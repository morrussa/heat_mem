# services/waypoint.py
import threading
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from ..config import Config
from ..models import MemoryItem, WaypointEdge  # 需要先在models.py中定义
from ..utils import compute_cosine_similarity


class WaypointService:
    """联想图服务：管理记忆之间的有向边，支持联想扩散"""
    
    def __init__(self, memory_module):
        self.memory_module = memory_module
        self.config: Config = memory_module.config
        self.lock = threading.RLock()
        # 可选：在内存中缓存边的出边列表，加速查询
        self.outgoing_cache: Dict[str, List[Tuple[str, float, int]]] = {}  # source_id -> [(target_id, weight, last_updated_turn)]
        self.cache_lock = threading.RLock()
        self._load_edges_from_db()
    
    def _load_edges_from_db(self):
        """启动时从数据库加载所有边到缓存"""
        cursor = self.memory_module.cursor
        cursor.execute(f"""
            SELECT source_id, target_id, weight, created_turn, last_updated_turn
            FROM {self.config.WAYPOINT_TABLE}
        """)
        rows = cursor.fetchall()
        with self.cache_lock:
            self.outgoing_cache.clear()
            for row in rows:
                source = row['source_id']
                target = row['target_id']
                weight = row['weight']
                last_turn = row['last_updated_turn']
                self.outgoing_cache.setdefault(source, []).append((target, weight, last_turn))
        print(f"[Waypoint] Loaded {len(rows)} edges from DB.")
    
    def add_or_update_edge(self, source_id: str, target_id: str, 
                           weight_delta: float = None, new_weight: float = None,
                           operation: str = 'reinforce'):
        """
        添加或更新一条边。
        - weight_delta: 增量更新（用于强化）
        - new_weight: 直接设置新权重（用于共现/语义）
        - operation: 记录来源，用于调试
        """
        if source_id == target_id:
            return  # 不建立自环
        
        with self.lock:
            cursor = self.memory_module.cursor
            current_turn = self.memory_module.current_turn
            
            # 查询现有边
            cursor.execute(f"""
                SELECT weight, created_turn FROM {self.config.WAYPOINT_TABLE}
                WHERE source_id = ? AND target_id = ?
            """, (source_id, target_id))
            row = cursor.fetchone()
            
            if row:
                old_weight = row['weight']
                if new_weight is not None:
                    final_weight = new_weight
                elif weight_delta is not None:
                    final_weight = old_weight + weight_delta
                else:
                    final_weight = old_weight  # 无变化则不更新
                
                # 权重衰减：若权重低于阈值则删除边
                if final_weight < self.config.WAYPOINT_MIN_WEIGHT:
                    cursor.execute(f"""
                        DELETE FROM {self.config.WAYPOINT_TABLE}
                        WHERE source_id = ? AND target_id = ?
                    """, (source_id, target_id))
                    self._remove_from_cache(source_id, target_id)
                    return
                
                # 更新边
                cursor.execute(f"""
                    UPDATE {self.config.WAYPOINT_TABLE}
                    SET weight = ?, last_updated_turn = ?
                    WHERE source_id = ? AND target_id = ?
                """, (final_weight, current_turn, source_id, target_id))
                self._update_cache(source_id, target_id, final_weight, current_turn)
            else:
                # 新建边
                if new_weight is not None:
                    init_weight = new_weight
                elif weight_delta is not None:
                    init_weight = weight_delta  # 若没有旧边，增量作为初始值
                else:
                    init_weight = self.config.WAYPOINT_DEFAULT_WEIGHT
                
                if init_weight >= self.config.WAYPOINT_MIN_WEIGHT:
                    cursor.execute(f"""
                        INSERT INTO {self.config.WAYPOINT_TABLE}
                        (source_id, target_id, weight, created_turn, last_updated_turn)
                        VALUES (?, ?, ?, ?, ?)
                    """, (source_id, target_id, init_weight, current_turn, current_turn))
                    self._add_to_cache(source_id, target_id, init_weight, current_turn)
    
    def _add_to_cache(self, source: str, target: str, weight: float, last_turn: int):
        with self.cache_lock:
            self.outgoing_cache.setdefault(source, []).append((target, weight, last_turn))
    
    def _update_cache(self, source: str, target: str, weight: float, last_turn: int):
        with self.cache_lock:
            edges = self.outgoing_cache.get(source)
            if edges:
                for i, (t, w, lt) in enumerate(edges):
                    if t == target:
                        edges[i] = (target, weight, last_turn)
                        break
    
    def _remove_from_cache(self, source: str, target: str):
        with self.cache_lock:
            edges = self.outgoing_cache.get(source)
            if edges:
                self.outgoing_cache[source] = [(t, w, lt) for t, w, lt in edges if t != target]
                if not self.outgoing_cache[source]:
                    del self.outgoing_cache[source]
    
    def get_outgoing_edges(self, source_id: str, current_turn: int, 
                           max_hops: int = 1, max_results: int = None,
                           decay_func=None) -> List[Tuple[str, float]]:
        """
        获取从 source_id 出发的联想记忆，支持多跳，并应用时间衰减。
        返回 (target_id, decayed_weight) 列表，按衰减后权重降序。
        decay_func: 自定义衰减函数，默认为指数衰减：weight * (decay_factor^(turns_elapsed))
        """
        if decay_func is None:
            decay_factor = self.config.WAYPOINT_DECAY_FACTOR
            def decay(w, last_turn):
                turns_elapsed = current_turn - last_turn
                return w * (decay_factor ** max(0, turns_elapsed))
        
        # 1-hop 邻居
        with self.cache_lock:
            edges = self.outgoing_cache.get(source_id, [])
        
        candidates = {}
        for target, weight, last_turn in edges:
            decayed = decay(weight, last_turn)
            if decayed > self.config.WAYPOINT_MIN_DECAYED_WEIGHT:
                candidates[target] = decayed
        
        if max_hops >= 2:
            # 2-hop 邻居：遍历每个1-hop邻居的出边
            two_hop_candidates = {}
            for target1 in list(candidates.keys()):
                with self.cache_lock:
                    edges2 = self.outgoing_cache.get(target1, [])
                for target2, weight2, last_turn2 in edges2:
                    if target2 == source_id:
                        continue  # 避免回环
                    # 路径权重 = 第一跳权重 * 第二跳权重
                    path_weight = candidates[target1] * decay(weight2, last_turn2)
                    if path_weight > self.config.WAYPOINT_MIN_DECAYED_WEIGHT:
                        if target2 not in two_hop_candidates or path_weight > two_hop_candidates[target2]:
                            two_hop_candidates[target2] = path_weight
            # 合并并去重
            for t, w in two_hop_candidates.items():
                if t not in candidates or w > candidates[t]:
                    candidates[t] = w
        
        # 排序
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        if max_results:
            sorted_candidates = sorted_candidates[:max_results]
        return sorted_candidates
    
    def batch_get_outgoing_edges(self, source_ids: List[str], current_turn: int,
                                 max_hops: int = 1, max_results_per_source: int = None,
                                 global_max: int = None) -> Dict[str, List[Tuple[str, float]]]:
        """批量查询，返回每个source的联想结果"""
        result = {}
        for sid in source_ids:
            edges = self.get_outgoing_edges(sid, current_turn, max_hops, max_results_per_source)
            if edges:
                result[sid] = edges
        if global_max:
            # 如果需要全局限制，可以合并所有结果后取top
            pass
        return result
    
    def reinforce_edges_from_query(self, seed_ids: List[str], hit_ids: List[str]):
        """
        根据一次查询中，种子记忆和扩散命中的记忆，强化种子->扩散的边。
        hit_ids 是扩散命中（包括可能的多跳）的记忆ID。
        通常对每个 seed_id，对每个 hit_id 增加一个小的增量。
        """
        delta = self.config.WAYPOINT_REINFORCE_DELTA
        for src in seed_ids:
            for tgt in hit_ids:
                if src != tgt:
                    self.add_or_update_edge(src, tgt, weight_delta=delta, operation='reinforce')
    
    def add_cooccurrence_edges(self, memory_ids: List[str], turn: int):
        """
        处理一轮对话中出现的所有记忆，增加它们之间的共现关联。
        通常建立双向边，权重基于共现次数（可使用log(count)）。
        此处简化：每对记忆增加一个固定增量，并归一化。
        """
        if len(memory_ids) < 2:
            return
        # 去重
        ids = list(set(memory_ids))
        # 对所有组合增加边
        increment = self.config.WAYPOINT_COOCCUR_INCREMENT
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                src, tgt = ids[i], ids[j]
                # 双向建立
                self.add_or_update_edge(src, tgt, weight_delta=increment, operation='cooccur')
                self.add_or_update_edge(tgt, src, weight_delta=increment, operation='cooccur')
    
    def add_semantic_edges_for_new_memory(self, new_memory: MemoryItem, candidates: List[MemoryItem]):
        """
        新记忆加入时，与一批候选记忆计算相似度，若在阈值区间内则添加弱边。
        """
        low = self.config.WAYPOINT_SEMANTIC_LOW
        high = self.config.WAYPOINT_SEMANTIC_HIGH
        init_weight = self.config.WAYPOINT_SEMANTIC_INIT_WEIGHT
        new_vec = new_memory.vector
        for mem in candidates:
            sim = compute_cosine_similarity(new_vec, mem.vector)
            if low < sim < high:
                # 双向建立弱边
                self.add_or_update_edge(new_memory.id, mem.id, new_weight=init_weight, operation='semantic')
                self.add_or_update_edge(mem.id, new_memory.id, new_weight=init_weight, operation='semantic')
    
    def get_stats(self) -> Dict[str, Any]:
        with self.cache_lock:
            return {
                'total_edges': sum(len(v) for v in self.outgoing_cache.values()),
                'unique_sources': len(self.outgoing_cache),
            }