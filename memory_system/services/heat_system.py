# services/heat_system.py
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ..config import Config
from ..models import MemoryItem, SemanticCluster
from ..utils import (
    update_memory_heat_in_db,
    update_cluster_heat_in_db,
    vector_to_blob,
    blob_to_vector,
    compute_cosine_similarity,
)
from ..infrastructure.locking import DistributedLockManager


class HeatSystem:
    """
    热力管理系统，负责记忆热力的分配、回收、暂存唤醒等操作。
    """

    def __init__(self, memory_module):
        self.memory_module = memory_module
        self.config: Config = memory_module.config
        self.lock_manager: DistributedLockManager = memory_module.lock_manager
        self.heat_pool_lock = threading.RLock()

        # 注意：原 pending_heat_per_cluster 字典已被移除，改用数据库表 pending_heat_units

    # ==================== 统一热力更新方法 ====================

    def _unified_update_heat(
        self,
        memory_id: str,
        new_heat: int,
        old_heat: int = None,
        cluster_id: str = None,
        update_memory: bool = True,
        update_cluster: bool = True,
        update_pool: bool = False,
        pool_delta: int = 0,
        adjust_unallocated: bool = True,
        tx: Optional["TransactionContext"] = None,
    ) -> bool:
        """
        统一热力更新方法 - 支持外部事务。
        （此方法与原版基本相同，但移除了对 pending_heat 的引用，因为已无内存字典）
        """
        from ..core import TransactionContext  # 避免循环导入

        memory = self.memory_module.hot_memories.get(memory_id)
        if not memory:
            memory = self.memory_module.sleeping_memories.get(memory_id)

        if not memory and old_heat is None:
            old_heat = 0

        if old_heat is None and memory:
            old_heat = memory.heat

        heat_delta = new_heat - old_heat

        is_new_memory = (
            memory_id not in self.memory_module.hot_memories
            and memory_id not in self.memory_module.sleeping_memories
        )

        if tx is not None:
            # 使用外部事务
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

        else:
            # 无外部事务时，使用内部事务
            with TransactionContext(self.memory_module) as inner_tx:
                if update_memory:
                    if memory:
                        memory.heat = new_heat
                        memory.update_count += 1
                    inner_tx.add_memory_heat_update(memory_id, old_heat, new_heat, cluster_id)
                    update_memory_heat_in_db(
                        self.memory_module.cursor,
                        self.config.MEMORY_TABLE,
                        memory_id,
                        new_heat,
                    )

                if update_cluster and cluster_id and heat_delta != 0:
                    if cluster_id in self.memory_module.clusters:
                        cluster = self.memory_module.clusters[cluster_id]
                        with cluster.lock:
                            cluster.total_heat += heat_delta
                            if cluster.total_heat < 0:
                                cluster.total_heat = 0
                        inner_tx.add_cluster_heat_update(cluster_id, heat_delta)
                    update_cluster_heat_in_db(
                        self.memory_module.cursor,
                        self.config.CLUSTER_TABLE,
                        cluster_id,
                        heat_delta,
                    )

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

                        # 从数据库统计 pending 热力（用于 unallocated 计算）
                        total_pending = self._get_total_pending_heat()

                        self.memory_module.unallocated_heat = max(
                            0,
                            self.config.TOTAL_HEAT
                            - total_memory_heat
                            - self.memory_module.heat_pool
                            - total_pending,
                        )
                        inner_tx.add_pool_update(pool_change)
                        self.memory_module.cursor.execute(
                            f"""
                            UPDATE {self.config.HEAT_POOL_TABLE}
                            SET heat_pool = ?, unallocated_heat = ?
                            WHERE id = 1
                        """,
                            (self.memory_module.heat_pool, self.memory_module.unallocated_heat),
                        )
                elif adjust_unallocated:
                    with self.heat_pool_lock:
                        total_memory_heat = 0
                        for mem in self.memory_module.hot_memories.values():
                            total_memory_heat += mem.heat
                        for mem in self.memory_module.sleeping_memories.values():
                            total_memory_heat += mem.heat
                        if is_new_memory:
                            total_memory_heat += new_heat

                        total_pending = self._get_total_pending_heat()

                        self.memory_module.unallocated_heat = max(
                            0,
                            self.config.TOTAL_HEAT
                            - total_memory_heat
                            - self.memory_module.heat_pool
                            - total_pending,
                        )
                        self.memory_module.cursor.execute(
                            f"""
                            UPDATE {self.config.HEAT_POOL_TABLE}
                            SET unallocated_heat = ?
                            WHERE id = 1
                        """,
                            (self.memory_module.unallocated_heat,),
                        )
                # inner_tx 退出时自动提交

        self.memory_module._invalidate_related_caches(memory_id, cluster_id)
        self.memory_module.operation_count += 1
        self.memory_module._trigger_maintenance_if_needed()
        return True

    # ==================== 热力池与回收 ====================

    def _update_unallocated_heat(self) -> int:
        """更新未分配热力，从数据库统计暂存热力"""
        with self.heat_pool_lock:
            total_memory_heat = sum(m.heat for m in self.memory_module.hot_memories.values()) + sum(
                m.heat for m in self.memory_module.sleeping_memories.values()
            )
            total_pending = self._get_total_pending_heat()

            total_allocated = self.memory_module.heat_pool + total_memory_heat + total_pending
            self.memory_module.unallocated_heat = max(0, self.config.TOTAL_HEAT - total_allocated)

            self.memory_module.cursor.execute(
                f"""
                UPDATE {self.config.HEAT_POOL_TABLE}
                SET unallocated_heat = ?
                WHERE id = 1
            """,
                (self.memory_module.unallocated_heat,),
            )

        return self.memory_module.unallocated_heat

    def _get_total_pending_heat(self) -> int:
        """从数据库查询所有暂存单元的热力总和"""
        try:
            cursor = self.memory_module.cursor
            cursor.execute(
                f"SELECT SUM(pending_heat) as total FROM {self.config.PENDING_HEAT_UNITS_TABLE} WHERE status='pending'"
            )
            row = cursor.fetchone()
            return row["total"] if row and row["total"] else 0
        except Exception as e:
            print(f"[HeatSystem] 查询暂存热力总和失败: {e}")
            return 0

    def _recycle_heat_pool(self, tx: Optional["TransactionContext"] = None):
        """回收热力到热力池"""
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
                        self.memory_module.cursor.execute(
                            f"""
                            UPDATE {self.config.HEAT_POOL_TABLE}
                            SET heat_pool = ?, unallocated_heat = ?
                            WHERE id = 1
                        """,
                            (self.memory_module.heat_pool, self.memory_module.unallocated_heat),
                        )

            self.memory_module.stats["heat_recycled_to_pool"] += transfer
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
                        self.memory_module.cursor.execute(
                            f"""
                            UPDATE {self.config.HEAT_POOL_TABLE}
                            SET heat_pool = ?, unallocated_heat = ?
                            WHERE id = 1
                        """,
                            (self.memory_module.heat_pool, self.memory_module.unallocated_heat),
                        )

            self.memory_module.stats["heat_recycled_to_pool"] += transfer

        if current_need > 0:
            self._recycle_from_memories(current_need, tx=tx)

    def _recycle_from_memories(self, need_heat: int, tx: Optional["TransactionContext"] = None):
        """从记忆中回收热力"""
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
                tx=tx,
            )
            recycled += to_recycle

        self.memory_module.stats["total_heat_recycled"] += recycled
        self._update_unallocated_heat()

    # ==================== 冷主导簇暂存热力 ====================

    def allocate_heat_with_cold_dominant(
        self,
        cluster_id: str,
        allocated_heat: int,
        best_similarity: float,
        tx: "TransactionContext",
        new_memory_id: str,
        new_memory_vector: np.ndarray,
    ) -> Tuple[int, bool]:
        """
        冷主导簇热力分配：将一半热力暂存到 pending_heat_units 表中，与新的记忆向量绑定。
        返回新记忆实际获得的热力，以及是否跳过邻居分配。
        """
        cluster = self.memory_module.clusters.get(cluster_id)
        if not cluster or best_similarity < self.config.CLUSTER_SIMILARITY_THRESHOLD:
            return allocated_heat, False

        # 判断是否为冷主导簇
        is_cold_dominant = (
            cluster.hot_memory_count < self.config.COLD_DOMINANT_HOT_COUNT_THRESHOLD
            and cluster.cold_memory_count >= self.config.COLD_DOMINANT_COLD_COUNT_MIN
        )
        if not is_cold_dominant:
            return allocated_heat, False

        # 暂存一半热力
        pending = allocated_heat // 2
        new_memory_final_heat = allocated_heat - pending

        # 通过事务插入暂存单元
        tx.add_pending_heat_unit(
            memory_id=new_memory_id,
            vector=new_memory_vector,
            pending_heat=pending,
            created_turn=self.memory_module.current_turn,
        )

        print(
            f"[HeatSystem] 冷主导簇 {cluster_id[:8]}...，暂存 {pending} 热力关联记忆 {new_memory_id[:8]}"
        )
        return new_memory_final_heat, True

    # ==================== 暂存单元处理 ====================

    def process_pending_heat_units(self):
        """
        定期维护任务：处理所有待唤醒的暂存单元。
        每个单元用其向量通过 Annoy 找到最匹配的冷簇，唤醒该簇内最相似的若干条冷记忆，并按相似度加权分配热力。
        """
        cursor = self.memory_module.cursor
        # 分批处理，每次最多处理 100 个单元（可配置）
        cursor.execute(
            f"""
            SELECT id, vector, pending_heat, created_turn
            FROM {self.config.PENDING_HEAT_UNITS_TABLE}
            WHERE status = 'pending'
            LIMIT 100
        """
        )
        rows = cursor.fetchall()
        if not rows:
            return

        from ..utils import blob_to_vector, compute_cosine_similarity
        from ..core import TransactionContext

        with TransactionContext(self.memory_module) as tx:
            for row in rows:
                mem_id = row["id"]
                vector = blob_to_vector(row["vector"])
                pending_heat = row["pending_heat"]
                created_turn = row["created_turn"]

                # 1. 用向量查找最近的簇
                best_cluster_id, best_sim = self.memory_module.cluster_service._find_best_cluster_annoy(
                    vector
                )
                if best_cluster_id is None or best_sim < self.config.CLUSTER_SIMILARITY_THRESHOLD:
                    # 无合适簇，热力归还热力池
                    tx.add_pool_update(pending_heat)
                    self._mark_pending_unit_done(tx, mem_id)
                    continue

                # 2. 从该簇的冷区记忆中加载候选记忆（多取一些用于排序）
                cold_memories = self.memory_module.cluster_service.load_cold_memories_from_cluster(
                    best_cluster_id,
                    limit=self.config.WAKEUP_MEMORIES_PER_CLUSTER
                    * 2,  # 例如 10 条
                )
                if not cold_memories:
                    # 无冷记忆，热力归还
                    tx.add_pool_update(pending_heat)
                    self._mark_pending_unit_done(tx, mem_id)
                    continue

                # 3. 计算每个冷记忆与暂存向量的相似度，排序取 Top-K
                candidates = []
                for cm in cold_memories:
                    cm_vector = blob_to_vector(cm["vector"])
                    sim = compute_cosine_similarity(vector, cm_vector)
                    candidates.append((cm, sim))
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_k = candidates[: self.config.WAKEUP_MEMORIES_PER_CLUSTER]  # 例如 5 条

                # 4. 按相似度加权分配热力
                if len(top_k) > 0:
                    sim_sum = sum(s for _, s in top_k)
                    for cm, sim in top_k:
                        heat_for_memory = (
                            int(pending_heat * (sim / sim_sum)) if sim_sum > 0 else pending_heat // len(top_k)
                        )
                        if heat_for_memory <= 0:
                            continue
                        self._wake_up_memory(cm, heat_for_memory, best_cluster_id, tx)
                else:
                    tx.add_pool_update(pending_heat)

                # 5. 标记暂存单元为已完成
                self._mark_pending_unit_done(tx, mem_id)

            # 事务提交
            self.memory_module.conn.commit()

        # 更新统计信息
        self.memory_module.stats["pending_heat_units"] = len(rows) - len(rows)  # 可重新查询
        self._update_unallocated_heat()

    def _wake_up_memory(
        self,
        cold_memory_data: Dict,
        heat_to_add: int,
        cluster_id: str,
        tx: "TransactionContext",
    ):
        """将冷记忆加入热区，分配热力"""
        memory = self._build_memory_from_db_row(cold_memory_data)
        if not memory:
            return
        # 将记忆加入热区
        self.memory_module.hot_memories[memory.id] = memory
        self.memory_module.memory_to_cluster[memory.id] = cluster_id

        # 更新记忆热力
        tx.add_memory_heat_update(
            memory_id=memory.id,
            old_heat=0,
            new_heat=heat_to_add,
            cluster_id=cluster_id,
        )

        # 更新簇统计
        cluster = self.memory_module.clusters.get(cluster_id)
        if cluster:
            with cluster.lock:
                cluster.hot_memory_count += 1
                cluster.cold_memory_count -= 1
                cluster.total_heat += heat_to_add
            tx.add_cluster_heat_update(cluster_id, heat_to_add)

        # 更新访问频率
        self.memory_module._update_access_frequency(memory.id)

    def _mark_pending_unit_done(self, tx: "TransactionContext", mem_id: str):
        """删除暂存单元"""
        tx.delete_pending_heat_unit(mem_id)

    def _build_memory_from_db_row(self, row: Dict) -> Optional[MemoryItem]:
        """从数据库行数据构建 MemoryItem 对象（与旧版相同）"""
        try:
            from ..models import MemoryItem
            from ..utils import blob_to_vector

            return MemoryItem(
                id=row["id"],
                vector=blob_to_vector(row["vector"]),
                user_input=row["user_input"],
                ai_response=row["ai_response"],
                summary=row.get("summary", ""),
                heat=row.get("heat", 0),
                created_turn=row.get("created_turn", 0),
                last_interaction_turn=row.get("last_interaction_turn", 0),
                access_count=row.get("access_count", 0),
                is_hot=row.get("is_hot", False),
                is_sleeping=row.get("is_sleeping", False),
                cluster_id=row.get("cluster_id"),
                metadata=row.get("metadata", {}),
                parent_turn=row.get("parent_turn"),
            )
        except Exception as e:
            print(f"[HeatSystem] 构建 MemoryItem 失败: {e}")
            return None

    # ==================== 热力分布调整 ====================

    def _check_and_adjust_heat_distribution(self):
        """检查并调整热力分布（与原版相同，但无 pending_heat 影响）"""
        from ..core import TransactionContext

        self.memory_module.maintenance_cycles_since_heat_check += 1
        if (
            self.memory_module.maintenance_cycles_since_heat_check
            < self.config.HEAT_RECYCLE_CHECK_FREQUENCY
        ):
            return
        self.memory_module.maintenance_cycles_since_heat_check = 0

        cluster_heat_list = []
        total_cluster_heat = 0
        for cluster_id, cluster in self.memory_module.clusters.items():
            if cluster.total_heat > 0:
                cluster_heat_list.append(
                    {"cluster_id": cluster_id, "heat": cluster.total_heat, "size": cluster.size}
                )
                total_cluster_heat += cluster.total_heat
                if cluster_id not in self.memory_module.cluster_heat_history:
                    self.memory_module.cluster_heat_history[cluster_id] = []
                self.memory_module.cluster_heat_history[cluster_id].append(
                    (self.memory_module.current_turn, cluster.total_heat)
                )

        if total_cluster_heat == 0 or len(cluster_heat_list) <= 5:
            return

        cluster_heat_list.sort(key=lambda x: x["heat"], reverse=True)
        top3_heat = sum(cluster["heat"] for cluster in cluster_heat_list[:3])
        top5_heat = sum(cluster["heat"] for cluster in cluster_heat_list[:5])
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
        self.memory_module.stats["heat_redistributions"] = (
            self.memory_module.stats.get("heat_redistributions", 0) + 1
        )

    def _redistribute_cluster_heat(self, cluster_heat_list: List[Dict], total_cluster_heat: int):
        """重新分配簇热力（与原版相同，但无 pending_heat 引用）"""
        from ..core import TransactionContext

        total_size = sum(cluster["size"] for cluster in cluster_heat_list)
        if total_size == 0:
            return

        top3_excess = max(
            0,
            sum(cluster["heat"] for cluster in cluster_heat_list[:3])
            - total_cluster_heat * self.config.TOP3_HEAT_LIMIT_RATIO,
        )
        top5_excess = max(
            0,
            sum(cluster["heat"] for cluster in cluster_heat_list[:5])
            - total_cluster_heat * self.config.TOP5_HEAT_LIMIT_RATIO,
        )
        excess_heat = max(top3_excess, top5_excess)

        if excess_heat <= 0:
            return

        total_top_heat = sum(cluster["heat"] for cluster in cluster_heat_list[:5])
        if total_top_heat == 0:
            return

        recycled_heat = 0

        with TransactionContext(self.memory_module) as tx:
            for i, cluster_info in enumerate(cluster_heat_list[:5]):
                cluster_id = cluster_info["cluster_id"]
                cluster = self.memory_module.clusters.get(cluster_id)
                if not cluster:
                    continue
                cluster_excess_ratio = cluster.total_heat / total_top_heat
                cluster_excess_heat = int(
                    excess_heat * cluster_excess_ratio * self.config.HEAT_RECYCLE_RATE
                )
                min_heat_for_cluster = max(
                    self.config.MIN_CLUSTER_HEAT_AFTER_RECYCLE, cluster.size * 10
                )
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

                    tx.add_memory_heat_update(memory.id, memory.heat, new_heat, cluster_id)
                    memory.heat = new_heat
                    memory.update_count += 1
                    recycled_heat += heat_to_deduct

                cluster.total_heat -= cluster_excess_heat
                tx.add_cluster_heat_update(cluster_id, -cluster_excess_heat)

            if recycled_heat > 0:
                with self.heat_pool_lock:
                    self.memory_module.heat_pool += recycled_heat
                    tx.add_pool_update(recycled_heat)
                self.memory_module.stats["heat_recycled_to_pool"] = (
                    self.memory_module.stats.get("heat_recycled_to_pool", 0) + recycled_heat
                )

        for cluster_info in cluster_heat_list[:5]:
            cluster_id = cluster_info["cluster_id"]
            self.memory_module.cache_manager.cluster_search_cache.clear(cluster_id)
        self.memory_module.cache_manager.invalidate_vector_cache()

    # ==================== 辅助方法 ====================

    def _is_in_suppression_period(self) -> bool:
        """检查是否处于热力回收抑制期"""
        if self.memory_module.last_heat_recycle_turn == 0:
            return False
        turns_since_recycle = (
            self.memory_module.current_turn - self.memory_module.last_heat_recycle_turn
        )
        return turns_since_recycle < self.config.HEAT_RECYCLE_SUPPRESSION_TURNS

    def _get_suppression_factor(self) -> float:
        """获取热力分配抑制因子"""
        if not self._is_in_suppression_period():
            return 1.0
        turns_since_recycle = (
            self.memory_module.current_turn - self.memory_module.last_heat_recycle_turn
        )
        remaining_suppression = max(
            0, self.config.HEAT_RECYCLE_SUPPRESSION_TURNS - turns_since_recycle
        )
        suppression_factor = self.config.HEAT_SUPPRESSION_FACTOR + (
            (1.0 - self.config.HEAT_SUPPRESSION_FACTOR)
            * (1.0 - remaining_suppression / self.config.HEAT_RECYCLE_SUPPRESSION_TURNS)
        )
        return min(1.0, max(self.config.HEAT_SUPPRESSION_FACTOR, suppression_factor))

    def _audit_heat_balance(self) -> Dict[str, Any]:
        """审计热力平衡（从数据库统计 pending 热力）"""
        total_hot_heat = sum(m.heat for m in self.memory_module.hot_memories.values())
        total_sleeping_heat = sum(m.heat for m in self.memory_module.sleeping_memories.values())
        total_cold_heat = 0

        total_pending = self._get_total_pending_heat()

        total_in_system = (
            self.memory_module.heat_pool
            + self.memory_module.unallocated_heat
            + total_hot_heat
            + total_sleeping_heat
            + total_cold_heat
            + total_pending
        )

        expected_total = self.config.TOTAL_HEAT
        discrepancy = total_in_system - expected_total

        audit_result = {
            "heat_pool": self.memory_module.heat_pool,
            "unallocated_heat": self.memory_module.unallocated_heat,
            "hot_memories_heat": total_hot_heat,
            "sleeping_memories_heat": total_sleeping_heat,
            "cold_memories_heat": total_cold_heat,
            "pending_heat": total_pending,
            "total_in_system": total_in_system,
            "expected_total": expected_total,
            "discrepancy": discrepancy,
            "has_leak": abs(discrepancy) > 100,
            "hot_memory_count": len(self.memory_module.hot_memories),
            "sleeping_memory_count": len(self.memory_module.sleeping_memories),
            "current_turn": self.memory_module.current_turn,
        }

        if abs(discrepancy) > 100:
            if discrepancy > 0:
                self.memory_module.unallocated_heat = max(
                    0, self.memory_module.unallocated_heat - discrepancy
                )
            else:
                self.memory_module.unallocated_heat += abs(discrepancy)
            self.memory_module.cursor.execute(
                f"""
                UPDATE {self.config.HEAT_POOL_TABLE}
                SET unallocated_heat = ?
                WHERE id = 1
            """,
                (self.memory_module.unallocated_heat,),
            )

        return audit_result

    def _update_cluster_heat(
        self,
        cluster_id: str,
        heat_delta: int,
        immediate: bool = True,
        tx: Optional["TransactionContext"] = None,
    ):
        """更新簇热力（与原版相同）"""
        from ..core import TransactionContext

        if immediate:
            if tx is not None:
                with self.lock_manager.with_lock(
                    f"cluster_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT
                ):
                    if cluster_id in self.memory_module.clusters:
                        self.memory_module.clusters[cluster_id].total_heat += heat_delta
                    tx.add_cluster_heat_update(cluster_id, heat_delta)
            else:
                with self.lock_manager.with_lock(
                    f"cluster_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT
                ):
                    if cluster_id in self.memory_module.clusters:
                        self.memory_module.clusters[cluster_id].total_heat += heat_delta
                    update_cluster_heat_in_db(
                        self.memory_module.cursor, self.config.CLUSTER_TABLE, cluster_id, heat_delta
                    )
        else:
            self.memory_module.update_queue.put(
                {
                    "type": "cluster_heat_update",
                    "cluster_id": cluster_id,
                    "heat_delta": heat_delta,
                    "turn": self.memory_module.current_turn,
                }
            )
        self.memory_module.cache_manager.weight_cache.clear()

    def get_pending_heat_stats(self) -> Dict[str, Any]:
        """获取暂存热力统计信息（从数据库查询）"""
        cursor = self.memory_module.cursor
        cursor.execute(
            f"SELECT COUNT(*) as cnt, SUM(pending_heat) as total FROM {self.config.PENDING_HEAT_UNITS_TABLE} WHERE status='pending'"
        )
        row = cursor.fetchone()
        return {
            "total_pending_heat": row["total"] if row and row["total"] else 0,
            "clusters_with_pending": row["cnt"] if row else 0,
        }

    def _save_pending_heat(self):
        """
        保存暂存热力状态（清理时调用）。由于暂存单元已在数据库中持久化，此方法可空实现或用于其他清理。
        保留此方法是为了兼容 MemoryModule.cleanup 中的调用，但无需额外操作。
        """
        # 暂存单元已在数据库中，无需额外保存
        pass