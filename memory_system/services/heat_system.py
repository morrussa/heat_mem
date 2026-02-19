import threading
from typing import Dict, List, Optional, Any, Tuple
from ..config import Config
from ..models import MemoryItem, SemanticCluster
from ..utils import update_memory_heat_in_db, update_cluster_heat_in_db, vector_to_blob, blob_to_vector
from ..infrastructure.locking import DistributedLockManager
import numpy as np
from collections import defaultdict


class HeatSystem:
    def __init__(self, memory_module):
        self.memory_module = memory_module
        self.config: Config = memory_module.config
        self.lock_manager: DistributedLockManager = memory_module.lock_manager
        self.heat_pool_lock = threading.RLock()
        
        # 暂存热力相关（用于冷主导簇）
        self.pending_heat_per_cluster: Dict[str, int] = {}
        self.pending_heat_lock = threading.RLock()

    def _unified_update_heat(self, memory_id: str, new_heat: int,
                            old_heat: int = None, cluster_id: str = None,
                            update_memory: bool = True, update_cluster: bool = True,
                            update_pool: bool = False, pool_delta: int = 0,
                            adjust_unallocated: bool = True,
                            tx: Optional['TransactionContext'] = None) -> bool:
        """统一热力更新方法 - 支持外部事务"""
        from ..core import TransactionContext  # 避免循环导入
        
        memory = self.memory_module.hot_memories.get(memory_id)
        if not memory:
            memory = self.memory_module.sleeping_memories.get(memory_id)

        if not memory and old_heat is None:
            old_heat = 0

        if old_heat is None and memory:
            old_heat = memory.heat

        heat_delta = new_heat - old_heat

        is_new_memory = (memory_id not in self.memory_module.hot_memories and
                         memory_id not in self.memory_module.sleeping_memories)

        if tx is not None:
            # 使用外部事务记录所有操作
            if update_memory:
                if memory:
                    memory.heat = new_heat
                    memory.update_count += 1
                tx.add_memory_heat_update(memory_id, old_heat, new_heat, cluster_id)

            if update_cluster and cluster_id and heat_delta != 0:
                if cluster_id in self.memory_module.clusters:
                    cluster = self.memory_module.clusters[cluster_id]
                    with cluster.lock:
                        cluster.total_heat += heat_delta
                        if cluster.total_heat < 0:
                            cluster.total_heat = 0
                tx.add_cluster_heat_update(cluster_id, heat_delta)

            if update_pool:
                tx.add_pool_update(pool_delta)

            # 注意：adjust_unallocated 在事务提交时通过 pool 更新自动处理
        else:
            # 无外部事务时，使用内部事务
            with TransactionContext(self.memory_module) as inner_tx:
                if update_memory:
                    if memory:
                        memory.heat = new_heat
                        memory.update_count += 1
                    inner_tx.add_memory_heat_update(memory_id, old_heat, new_heat, cluster_id)
                    update_memory_heat_in_db(self.memory_module.cursor, self.config.MEMORY_TABLE,
                                            memory_id, new_heat)

                if update_cluster and cluster_id and heat_delta != 0:
                    if cluster_id in self.memory_module.clusters:
                        cluster = self.memory_module.clusters[cluster_id]
                        with cluster.lock:
                            cluster.total_heat += heat_delta
                            if cluster.total_heat < 0:
                                cluster.total_heat = 0
                        inner_tx.add_cluster_heat_update(cluster_id, heat_delta)
                    update_cluster_heat_in_db(self.memory_module.cursor, self.config.CLUSTER_TABLE,
                                             cluster_id, heat_delta)

                if update_pool:
                    with self.heat_pool_lock:
                        pool_change = pool_delta
                        if self.memory_module.heat_pool + pool_change < 0:
                            pool_change = -self.memory_module.heat_pool
                        self.memory_module.heat_pool += pool_change
                        if self.memory_module.heat_pool < 0:
                            self.memory_module.heat_pool = 0
                        total_memory_heat = 0
                        for mem in self.memory_module.hot_memories.values():
                            total_memory_heat += mem.heat
                        for mem in self.memory_module.sleeping_memories.values():
                            total_memory_heat += mem.heat
                        if is_new_memory:
                            total_memory_heat += new_heat
                        
                        # 计入暂存热力
                        with self.pending_heat_lock:
                            total_pending = sum(self.pending_heat_per_cluster.values())
                        
                        self.memory_module.unallocated_heat = max(0, self.config.TOTAL_HEAT -
                                                                  total_memory_heat - 
                                                                  self.memory_module.heat_pool -
                                                                  total_pending)
                        inner_tx.add_pool_update(pool_change)  # 记录池子更新
                        self.memory_module.cursor.execute(f"""
                            UPDATE {self.config.HEAT_POOL_TABLE}
                            SET heat_pool = ?, unallocated_heat = ?
                            WHERE id = 1
                        """, (self.memory_module.heat_pool, self.memory_module.unallocated_heat))
                elif adjust_unallocated:
                    with self.heat_pool_lock:
                        total_memory_heat = 0
                        for mem in self.memory_module.hot_memories.values():
                            total_memory_heat += mem.heat
                        for mem in self.memory_module.sleeping_memories.values():
                            total_memory_heat += mem.heat
                        if is_new_memory:
                            total_memory_heat += new_heat
                        
                        # 计入暂存热力
                        with self.pending_heat_lock:
                            total_pending = sum(self.pending_heat_per_cluster.values())
                        
                        self.memory_module.unallocated_heat = max(0, self.config.TOTAL_HEAT -
                                                                  total_memory_heat - 
                                                                  self.memory_module.heat_pool -
                                                                  total_pending)
                        self.memory_module.cursor.execute(f"""
                            UPDATE {self.config.HEAT_POOL_TABLE}
                            SET unallocated_heat = ?
                            WHERE id = 1
                        """, (self.memory_module.unallocated_heat,))
                # inner_tx 退出时自动提交

        self.memory_module._invalidate_related_caches(memory_id, cluster_id)
        self.memory_module.operation_count += 1
        self.memory_module._trigger_maintenance_if_needed()
        return True

    def _update_unallocated_heat(self):
        """更新未分配热力，计入暂存热力"""
        with self.heat_pool_lock:
            total_memory_heat = sum(m.heat for m in self.memory_module.hot_memories.values()) + \
                                sum(m.heat for m in self.memory_module.sleeping_memories.values())
            
            with self.pending_heat_lock:
                total_pending = sum(self.pending_heat_per_cluster.values())
            
            total_allocated = self.memory_module.heat_pool + total_memory_heat + total_pending
            self.memory_module.unallocated_heat = max(0, self.config.TOTAL_HEAT - total_allocated)
            
            self.memory_module.cursor.execute(f"""
                UPDATE {self.config.HEAT_POOL_TABLE}
                SET unallocated_heat = ?
                WHERE id = 1
            """, (self.memory_module.unallocated_heat,))
        
        return self.memory_module.unallocated_heat

    def _recycle_heat_pool(self, tx: Optional['TransactionContext'] = None):
        """回收热力到热力池 - 支持事务传递"""
        from ..core import TransactionContext
        
        self.memory_module._unified_turn_increment()
        target_pool_size = self.config.INITIAL_HEAT_POOL
        current_need = target_pool_size - self.memory_module.heat_pool
        if current_need <= 0:
            return

        # 首先从未分配热力中提取
        if self.memory_module.unallocated_heat >= current_need:
            with self.heat_pool_lock:
                transfer = min(self.memory_module.unallocated_heat, current_need)
                self.memory_module.heat_pool += transfer
                self.memory_module.unallocated_heat -= transfer
                
                if tx is not None:
                    tx.add_pool_update(transfer)
                else:
                    with TransactionContext(self.memory_module) as inner_tx:
                        inner_tx.add_pool_update(transfer)
                        self.memory_module.cursor.execute(f"""
                            UPDATE {self.config.HEAT_POOL_TABLE}
                            SET heat_pool = ?, unallocated_heat = ?
                            WHERE id = 1
                        """, (self.memory_module.heat_pool, self.memory_module.unallocated_heat))
                        
            self.memory_module.stats['heat_recycled_to_pool'] += transfer
            return

        if self.memory_module.unallocated_heat > 0:
            with self.heat_pool_lock:
                transfer = self.memory_module.unallocated_heat
                self.memory_module.heat_pool += transfer
                self.memory_module.unallocated_heat = 0
                current_need -= transfer
                
                if tx is not None:
                    tx.add_pool_update(transfer)
                else:
                    with TransactionContext(self.memory_module) as inner_tx:
                        inner_tx.add_pool_update(transfer)
                        self.memory_module.cursor.execute(f"""
                            UPDATE {self.config.HEAT_POOL_TABLE}
                            SET heat_pool = ?, unallocated_heat = ?
                            WHERE id = 1
                        """, (self.memory_module.heat_pool, self.memory_module.unallocated_heat))
                        
            self.memory_module.stats['heat_recycled_to_pool'] += transfer

        if current_need > 0:
            self._recycle_from_memories(current_need, tx=tx)

    def _recycle_from_memories(self, need_heat: int, tx: Optional['TransactionContext'] = None):
        """从记忆中回收热力 - 支持事务传递"""
        from ..core import TransactionContext
        
        if need_heat <= 0:
            return

        eligible_memories = []
        for memory_id, memory in self.memory_module.hot_memories.items():
            if memory.heat > 10 and not memory.is_sleeping:
                eligible_memories.append((memory_id, memory))

        if not eligible_memories:
            return

        eligible_memories.sort(key=lambda x: x[1].heat, reverse=True)
        recycled = 0
        for memory_id, memory in eligible_memories:
            if recycled >= need_heat:
                break
            max_recyclable = max(0, memory.heat - 10)
            to_recycle = min(max_recyclable, need_heat - recycled)
            if to_recycle <= 0:
                continue
            new_heat = memory.heat - to_recycle
            self._unified_update_heat(
                memory_id=memory_id,
                new_heat=new_heat,
                old_heat=memory.heat,
                cluster_id=memory.cluster_id,
                update_memory=True,
                update_cluster=True,
                update_pool=True,
                pool_delta=to_recycle,
                tx=tx  # 传递事务
            )
            recycled += to_recycle
            
        self.memory_module.stats['total_heat_recycled'] += recycled
        self._update_unallocated_heat()

    def _check_and_adjust_heat_distribution(self):
        """检查并调整热力分布 - 使用内部事务"""
        from ..core import TransactionContext
        
        self.memory_module.maintenance_cycles_since_heat_check += 1
        if self.memory_module.maintenance_cycles_since_heat_check < self.config.HEAT_RECYCLE_CHECK_FREQUENCY:
            return
        self.memory_module.maintenance_cycles_since_heat_check = 0

        cluster_heat_list = []
        total_cluster_heat = 0
        for cluster_id, cluster in self.memory_module.clusters.items():
            if cluster.total_heat > 0:
                cluster_heat_list.append({
                    'cluster_id': cluster_id,
                    'heat': cluster.total_heat,
                    'size': cluster.size
                })
                total_cluster_heat += cluster.total_heat
                if cluster_id not in self.memory_module.cluster_heat_history:
                    self.memory_module.cluster_heat_history[cluster_id] = []
                self.memory_module.cluster_heat_history[cluster_id].append((self.memory_module.current_turn, cluster.total_heat))

        if total_cluster_heat == 0 or len(cluster_heat_list) <= 5:
            return

        cluster_heat_list.sort(key=lambda x: x['heat'], reverse=True)
        top3_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:3])
        top5_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:5])
        top3_ratio = top3_heat / total_cluster_heat
        top5_ratio = top5_heat / total_cluster_heat

        need_recycle = False
        if top3_ratio > self.config.TOP3_HEAT_LIMIT_RATIO:
            need_recycle = True
        if top5_ratio > self.config.TOP5_HEAT_LIMIT_RATIO:
            need_recycle = True

        if not need_recycle:
            return

        self._redistribute_cluster_heat(cluster_heat_list, total_cluster_heat)
        self.memory_module.last_heat_recycle_turn = self.memory_module.current_turn
        self.memory_module.heat_recycle_count += 1
        self.memory_module.stats['heat_redistributions'] = self.memory_module.stats.get('heat_redistributions', 0) + 1

    def _redistribute_cluster_heat(self, cluster_heat_list: List[Dict], total_cluster_heat: int):
        """重新分配簇热力 - 使用事务确保原子性"""
        from ..core import TransactionContext
        
        total_size = sum(cluster['size'] for cluster in cluster_heat_list)
        if total_size == 0:
            return

        top3_excess = max(0, sum(cluster['heat'] for cluster in cluster_heat_list[:3]) -
                         total_cluster_heat * self.config.TOP3_HEAT_LIMIT_RATIO)
        top5_excess = max(0, sum(cluster['heat'] for cluster in cluster_heat_list[:5]) -
                         total_cluster_heat * self.config.TOP5_HEAT_LIMIT_RATIO)
        excess_heat = max(top3_excess, top5_excess)

        if excess_heat <= 0:
            return

        total_top_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:5])
        if total_top_heat == 0:
            return

        recycled_heat = 0
        
        # 使用单个事务包裹整个重分配过程
        with TransactionContext(self.memory_module) as tx:
            for i, cluster_info in enumerate(cluster_heat_list[:5]):
                cluster_id = cluster_info['cluster_id']
                cluster = self.memory_module.clusters.get(cluster_id)
                if not cluster:
                    continue
                cluster_excess_ratio = cluster.total_heat / total_top_heat
                cluster_excess_heat = int(excess_heat * cluster_excess_ratio * self.config.HEAT_RECYCLE_RATE)
                min_heat_for_cluster = max(self.config.MIN_CLUSTER_HEAT_AFTER_RECYCLE, cluster.size * 10)
                if cluster.total_heat - cluster_excess_heat < min_heat_for_cluster:
                    cluster_excess_heat = max(0, cluster.total_heat - min_heat_for_cluster)
                if cluster_excess_heat <= 0:
                    continue

                memories_to_adjust = []
                for memory_id in list(cluster.memory_ids):
                    if memory_id in self.memory_module.hot_memories:
                        memories_to_adjust.append(self.memory_module.hot_memories[memory_id])
                if not memories_to_adjust:
                    continue
                    
                heat_per_memory = max(1, cluster_excess_heat // len(memories_to_adjust))
                for memory in memories_to_adjust:
                    heat_to_deduct = min(heat_per_memory, memory.heat - 1)
                    if heat_to_deduct <= 0:
                        continue
                    new_heat = memory.heat - heat_to_deduct
                    
                    # 使用 tx 记录记忆热力更新
                    tx.add_memory_heat_update(memory.id, memory.heat, new_heat, cluster_id)
                    memory.heat = new_heat
                    memory.update_count += 1
                    recycled_heat += heat_to_deduct

                # 更新簇热力
                cluster.total_heat -= cluster_excess_heat
                tx.add_cluster_heat_update(cluster_id, -cluster_excess_heat)

            if recycled_heat > 0:
                # 将回收的热力加入热力池
                with self.heat_pool_lock:
                    self.memory_module.heat_pool += recycled_heat
                    tx.add_pool_update(recycled_heat)
                self.memory_module.stats['heat_recycled_to_pool'] = self.memory_module.stats.get('heat_recycled_to_pool', 0) + recycled_heat

        # 事务外清理缓存
        for cluster_info in cluster_heat_list[:5]:
            cluster_id = cluster_info['cluster_id']
            self.memory_module.cache_manager.cluster_search_cache.clear(cluster_id)
        self.memory_module.cache_manager.invalidate_vector_cache()

    def _is_in_suppression_period(self) -> bool:
        """检查是否处于热力回收抑制期"""
        if self.memory_module.last_heat_recycle_turn == 0:
            return False
        turns_since_recycle = self.memory_module.current_turn - self.memory_module.last_heat_recycle_turn
        return turns_since_recycle < self.config.HEAT_RECYCLE_SUPPRESSION_TURNS

    def _get_suppression_factor(self) -> float:
        """获取热力分配抑制因子"""
        if not self._is_in_suppression_period():
            return 1.0
        turns_since_recycle = self.memory_module.current_turn - self.memory_module.last_heat_recycle_turn
        remaining_suppression = max(0, self.config.HEAT_RECYCLE_SUPPRESSION_TURNS - turns_since_recycle)
        suppression_factor = self.config.HEAT_SUPPRESSION_FACTOR + (
            (1.0 - self.config.HEAT_SUPPRESSION_FACTOR) *
            (1.0 - remaining_suppression / self.config.HEAT_RECYCLE_SUPPRESSION_TURNS)
        )
        return min(1.0, max(self.config.HEAT_SUPPRESSION_FACTOR, suppression_factor))

    def _audit_heat_balance(self) -> Dict[str, Any]:
        """审计热力平衡"""
        total_hot_heat = sum(m.heat for m in self.memory_module.hot_memories.values())
        total_sleeping_heat = sum(m.heat for m in self.memory_module.sleeping_memories.values())
        total_cold_heat = 0
        
        with self.pending_heat_lock:
            total_pending = sum(self.pending_heat_per_cluster.values())
        
        total_in_system = (self.memory_module.heat_pool +
                          self.memory_module.unallocated_heat +
                          total_hot_heat +
                          total_sleeping_heat +
                          total_cold_heat +
                          total_pending)
        
        expected_total = self.config.TOTAL_HEAT
        discrepancy = total_in_system - expected_total
        
        audit_result = {
            'heat_pool': self.memory_module.heat_pool,
            'unallocated_heat': self.memory_module.unallocated_heat,
            'hot_memories_heat': total_hot_heat,
            'sleeping_memories_heat': total_sleeping_heat,
            'cold_memories_heat': total_cold_heat,
            'pending_heat': total_pending,
            'total_in_system': total_in_system,
            'expected_total': expected_total,
            'discrepancy': discrepancy,
            'has_leak': abs(discrepancy) > 100,
            'hot_memory_count': len(self.memory_module.hot_memories),
            'sleeping_memory_count': len(self.memory_module.sleeping_memories),
            'current_turn': self.memory_module.current_turn
        }
        
        # 如果有差异，调整 unallocated_heat
        if abs(discrepancy) > 100:
            if discrepancy > 0:
                self.memory_module.unallocated_heat = max(0, self.memory_module.unallocated_heat - discrepancy)
            else:
                self.memory_module.unallocated_heat += abs(discrepancy)
            self.memory_module.cursor.execute(f"""
                UPDATE {self.config.HEAT_POOL_TABLE}
                SET unallocated_heat = ?
                WHERE id = 1
            """, (self.memory_module.unallocated_heat,))
        
        return audit_result

    def _update_cluster_heat(self, cluster_id: str, heat_delta: int, immediate: bool = True,
                            tx: Optional['TransactionContext'] = None):
        """更新簇热力 - 支持事务传递"""
        from ..core import TransactionContext
        
        if immediate:
            if tx is not None:
                # 使用外部事务
                with self.lock_manager.with_lock(f"cluster_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT):
                    if cluster_id in self.memory_module.clusters:
                        self.memory_module.clusters[cluster_id].total_heat += heat_delta
                    tx.add_cluster_heat_update(cluster_id, heat_delta)
            else:
                # 无外部事务，直接更新数据库
                with self.lock_manager.with_lock(f"cluster_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT):
                    if cluster_id in self.memory_module.clusters:
                        self.memory_module.clusters[cluster_id].total_heat += heat_delta
                    update_cluster_heat_in_db(self.memory_module.cursor, self.config.CLUSTER_TABLE, cluster_id, heat_delta)
        else:
            self.memory_module.update_queue.put({
                'type': 'cluster_heat_update',
                'cluster_id': cluster_id,
                'heat_delta': heat_delta,
                'turn': self.memory_module.current_turn
            })
        self.memory_module.cache_manager.weight_cache.clear()

    def allocate_heat_with_cold_dominant(self, cluster_id: str, allocated_heat: int, 
                                         best_similarity: float, tx: 'TransactionContext') -> Tuple[int, bool]:
        """
        冷主导簇热力分配处理
        
        Args:
            cluster_id: 目标簇ID
            allocated_heat: 计划分配的总热力
            best_similarity: 与簇质心的相似度
            tx: 事务上下文
            
        Returns:
            Tuple[int, bool]: (新记忆实际获得的热力, 是否跳过邻居分配)
        """
        cluster = self.memory_module.clusters.get(cluster_id)
        if not cluster or best_similarity < self.config.CLUSTER_SIMILARITY_THRESHOLD:
            return allocated_heat, False

        # 判断是否为冷主导簇
        is_cold_dominant = (cluster.hot_memory_count < self.config.COLD_DOMINANT_HOT_COUNT_THRESHOLD 
                            and cluster.cold_memory_count >= self.config.COLD_DOMINANT_COLD_COUNT_MIN)
        
        if not is_cold_dominant:
            return allocated_heat, False

        # 冷主导簇：暂存一半热力用于唤醒冷区记忆
        total_neighbor_heat = allocated_heat // 2
        new_memory_final_heat = allocated_heat - total_neighbor_heat
        
        # 通过事务添加暂存热力
        tx.add_pending_heat_update(cluster_id, total_neighbor_heat)
        
        # 同时更新内存中的暂存热力（保持一致性）
        with self.pending_heat_lock:
            self.pending_heat_per_cluster[cluster_id] = (
                self.pending_heat_per_cluster.get(cluster_id, 0) + total_neighbor_heat
            )
        
        print(f"[HeatSystem] 检测到冷主导簇 {cluster_id[:8]}...，暂存 {total_neighbor_heat} 热力")
        
        return new_memory_final_heat, True

    def process_pending_cluster_heat(self):
        """处理所有暂存热力，唤醒冷区记忆（使用事务）"""
        from ..core import TransactionContext
        
        # 获取所有有暂存热力的簇（从数据库读，确保一致性）
        cursor = self.memory_module.cursor
        cursor.execute(f"""
            SELECT cluster_id, pending_heat 
            FROM {self.memory_module.config.PENDING_HEAT_TABLE}
            WHERE pending_heat > 0
        """)
        rows = cursor.fetchall()
        
        if not rows:
            return

        with TransactionContext(self.memory_module) as tx:
            for cluster_id, pending_heat in rows:
                if pending_heat <= 0:
                    continue
                
                # 从数据库加载该簇的冷区记忆
                cold_memories = self._load_cold_memories_from_cluster(cluster_id, 
                                    limit=self.config.WAKEUP_MEMORIES_PER_CLUSTER)
                if not cold_memories:
                    print(f"[HeatSystem] 簇 {cluster_id[:8]}... 没有可唤醒的冷区记忆")
                    continue

                # 计算每个记忆应得的热力
                heat_per_memory = pending_heat // len(cold_memories)
                remainder = pending_heat - heat_per_memory * len(cold_memories)
                
                cluster = self.memory_module.clusters.get(cluster_id)
                if not cluster:
                    continue

                for i, memory_data in enumerate(cold_memories):
                    memory_id = memory_data['id']
                    
                    # 从数据库加载记忆并转换为 MemoryItem
                    memory = self._build_memory_from_db_row(memory_data)
                    if not memory:
                        continue
                        
                    memory_heat = heat_per_memory + (1 if i < remainder else 0)
                    
                    # 通过事务添加记忆热力更新
                    tx.add_memory_heat_update(
                        memory_id=memory_id,
                        old_heat=0,
                        new_heat=memory_heat,
                        cluster_id=cluster_id
                    )
                    
                    # 加入热区
                    self.memory_module.hot_memories[memory_id] = memory
                    self.memory_module.memory_to_cluster[memory_id] = cluster_id
                    
                    # 更新簇统计信息
                    with cluster.lock:
                        cluster.hot_memory_count += 1
                        cluster.cold_memory_count -= 1
                        cluster.memory_ids.add(memory_id)
                        cluster.total_heat += memory_heat
                    
                    print(f"[HeatSystem] 唤醒记忆 {memory_id[:8]}...，分配 {memory_heat} 热力")

                # 清除暂存热力（通过事务）
                tx.add_pending_heat_update(cluster_id, -pending_heat)
                
                # 更新内存中的暂存热力
                with self.pending_heat_lock:
                    if cluster_id in self.pending_heat_per_cluster:
                        del self.pending_heat_per_cluster[cluster_id]
                
                # 更新簇信息到数据库
                self._update_cluster_in_db(cluster)
            
            # 事务提交时自动应用所有更新
            self.memory_module.conn.commit()
        
        self._update_unallocated_heat()

    def _load_cold_memories_from_cluster(self, cluster_id: str, limit: int = 5) -> List[Dict]:
        """从数据库加载指定簇的冷区记忆"""
        try:
            self.memory_module.cursor.execute(f"""
                SELECT * FROM {self.config.MEMORY_TABLE}
                WHERE cluster_id = ? AND is_hot = 0
                ORDER BY last_interaction_turn DESC, heat DESC
                LIMIT ?
            """, (cluster_id, limit))
            
            rows = self.memory_module.cursor.fetchall()
            memories = []
            for row in rows:
                memories.append({
                    'id': row['id'],
                    'vector': row['vector'],
                    'user_input': row['user_input'],
                    'ai_response': row['ai_response'],
                    'summary': row['summary'] or "",
                    'heat': row['heat'],
                    'created_turn': row['created_turn'],
                    'last_interaction_turn': row['last_interaction_turn'],
                    'access_count': row['access_count'],
                    'metadata': row['metadata'],
                    'parent_turn': row.get('parent_turn'),
                    'is_hot': bool(row['is_hot']),
                    'is_sleeping': bool(row['is_sleeping']),
                    'cluster_id': row['cluster_id']
                })
            return memories
        except Exception as e:
            print(f"[HeatSystem] 从簇 {cluster_id} 加载冷区记忆失败: {e}")
            return []

    def _build_memory_from_db_row(self, row: Dict) -> Optional[MemoryItem]:
        """从数据库行数据构建 MemoryItem 对象"""
        try:
            from ..models import MemoryItem
            from ..utils import blob_to_vector
            
            return MemoryItem(
                id=row['id'],
                vector=blob_to_vector(row['vector']),
                user_input=row['user_input'],
                ai_response=row['ai_response'],
                summary=row.get('summary', ''),
                heat=row.get('heat', 0),
                created_turn=row.get('created_turn', 0),
                last_interaction_turn=row.get('last_interaction_turn', 0),
                access_count=row.get('access_count', 0),
                is_hot=row.get('is_hot', False),
                is_sleeping=row.get('is_sleeping', False),
                cluster_id=row.get('cluster_id'),
                metadata=row.get('metadata', {}),
                parent_turn=row.get('parent_turn')
            )
        except Exception as e:
            print(f"[HeatSystem] 构建 MemoryItem 失败: {e}")
            return None

    def _update_cluster_in_db(self, cluster: SemanticCluster):
        """更新簇信息到数据库"""
        try:
            self.memory_module.cursor.execute(f"""
                UPDATE {self.config.CLUSTER_TABLE}
                SET total_heat = ?, hot_memory_count = ?, cold_memory_count = ?,
                    size = ?, last_updated_turn = ?, version = version + 1
                WHERE id = ?
            """, (
                cluster.total_heat,
                cluster.hot_memory_count,
                cluster.cold_memory_count,
                cluster.size,
                self.memory_module.current_turn,
                cluster.id
            ))
        except Exception as e:
            print(f"[HeatSystem] 更新簇 {cluster.id[:8]}... 到数据库失败: {e}")

    def get_pending_heat_stats(self) -> Dict[str, Any]:
        """获取暂存热力统计信息"""
        with self.pending_heat_lock:
            total_pending = sum(self.pending_heat_per_cluster.values())
            return {
                'total_pending_heat': total_pending,
                'clusters_with_pending': len(self.pending_heat_per_cluster),
                'pending_heat_per_cluster': dict(self.pending_heat_per_cluster)
            }

    def _save_pending_heat(self):
        """保存暂存热力状态（用于清理时）"""
        with self.pending_heat_lock:
            cursor = self.memory_module.cursor
            for cluster_id, heat in self.pending_heat_per_cluster.items():
                if heat > 0:
                    cursor.execute(f"""
                        INSERT INTO {self.memory_module.config.PENDING_HEAT_TABLE} 
                        (cluster_id, pending_heat, version, last_updated_turn)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(cluster_id) DO UPDATE SET
                            pending_heat = ?,
                            version = version + 1,
                            last_updated_turn = ?
                    """, (cluster_id, heat, self.memory_module.current_turn, heat, self.memory_module.current_turn))
            self.memory_module.conn.commit()