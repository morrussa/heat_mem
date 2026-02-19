# core.py
import sqlite3
import numpy as np
import json
import pickle
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
import warnings
import struct
import hashlib
from collections import defaultdict, deque
import heapq
from pathlib import Path
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import uuid

from .config import Config, OperationType, ConsistencyLevel
from .models import MemoryItem, SemanticCluster, WeightedMemoryResult, LayeredSearchResult, VectorCache
from .utils import (
    vector_to_blob, blob_to_vector, increment_turn, allocate_heat_from_pool,
    update_memory_heat_in_db, update_cluster_heat_in_db, invalidate_memory_caches,
    schedule_centroid_update, execute_with_retry, compute_cosine_similarity,
    compute_batch_similarities, convert_memory_vectors
)
from .infrastructure.database import Database
from .infrastructure.cache import CacheManager
from .infrastructure.locking import DistributedLockManager
from .infrastructure.history import HistoryManager
from .services.heat_system import HeatSystem
from .services.cluster_service import ClusterService
from .services.search_service import SearchService
# === Waypoint Integration: import new service ===
from .services.waypoint import WaypointService


# =============== 事务上下文 ===============
class TransactionContext:
    def __init__(self, memory_module, consistency_level: ConsistencyLevel = ConsistencyLevel.STRONG):
        self.memory_module = memory_module
        self.consistency_level = consistency_level
        self.operations = []
        self.memory_updates = {}
        self.cluster_updates = {}
        self.pending_heat_updates = {}
        self.pool_updates = 0  # 新增：记录热力池总变更
        self.transaction_id = str(uuid.uuid4())
        self.transaction_started = False

    def __enter__(self):
        if self.consistency_level == ConsistencyLevel.STRONG:
            self.transaction_started = self.memory_module._ensure_transaction(self.transaction_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            if self.consistency_level == ConsistencyLevel.STRONG:
                success = True
                for op in self.operations:
                    try:
                        self.memory_module._apply_operation(op, immediate=True)
                    except Exception as e:
                        print(f"Failed to apply operation: {e}")
                        success = False
                if self.transaction_started:
                    self.memory_module._finalize_transaction(self.transaction_id, success)
                    if not success:
                        raise Exception("Transaction failed")
            elif self.consistency_level == ConsistencyLevel.IMMEDIATE:
                self.memory_module._apply_immediate_updates(self.operations)
            else:
                self.memory_module._queue_eventual_updates(self.operations)
        else:
            if self.consistency_level == ConsistencyLevel.STRONG and self.transaction_started:
                self.memory_module._finalize_transaction(self.transaction_id, False)

    def add_memory_heat_update(self, memory_id: str, old_heat: int, new_heat: int, cluster_id: str = None):
        """添加记忆热力更新操作"""
        operation = {
            'type': OperationType.MEMORY_HEAT_UPDATE,
            'memory_id': memory_id,
            'old_heat': old_heat,
            'new_heat': new_heat,
            'cluster_id': cluster_id,
            'transaction_id': self.transaction_id
        }
        self.operations.append(operation)
        self.memory_updates[memory_id] = new_heat

    def add_cluster_heat_update(self, cluster_id: str, heat_delta: int):
        """添加簇热力更新操作"""
        operation = {
            'type': OperationType.CLUSTER_HEAT_UPDATE,
            'cluster_id': cluster_id,
            'heat_delta': heat_delta,
            'transaction_id': self.transaction_id
        }
        self.operations.append(operation)
        self.cluster_updates[cluster_id] = self.cluster_updates.get(cluster_id, 0) + heat_delta

    def add_pending_heat_update(self, cluster_id: str, delta: int):
        """添加暂存热力更新操作"""
        operation = {
            'type': OperationType.PENDING_HEAT_UPDATE,
            'cluster_id': cluster_id,
            'delta': delta,
            'transaction_id': self.transaction_id
        }
        self.operations.append(operation)
        self.pending_heat_updates[cluster_id] = self.pending_heat_updates.get(cluster_id, 0) + delta

    def add_pool_update(self, pool_delta: int):
        """添加热力池更新操作"""
        operation = {
            'type': OperationType.POOL_HEAT_UPDATE,
            'pool_delta': pool_delta,
            'transaction_id': self.transaction_id
        }
        self.operations.append(operation)
        self.pool_updates += pool_delta

    def add_atomic_memories(self, facts: List[str], user_input: str, ai_response: str,
                            metadata: Dict[str, Any] = None) -> List[str]:
        """将原子事实存储为独立的记忆项，并处理重复检测与合并"""
        if metadata is None:
            metadata = {}
        
        current_turn = self.memory_module.current_turn
        memory_ids = []
        
        if not facts:
            return memory_ids
        
        print(f"[Memory] 开始存储 {len(facts)} 条原子事实...")
        
        for i, fact in enumerate(facts):
            if not fact or len(fact.strip()) < 5:
                continue
                
            # 生成向量
            fact_vector = self.memory_module._get_embedding(fact)
            
            # ========== 新增：重复检测与合并 ==========
            duplicate_id = self.memory_module._check_duplicate(fact_vector, fact)
            if duplicate_id:
                print(f"[Memory] 发现重复原子事实，合并到已有记忆 {duplicate_id[:8]}...")
                
                # 获取原记忆对象（优先从内存，否则从数据库）
                original_memory = None
                if duplicate_id in self.memory_module.hot_memories:
                    original_memory = self.memory_module.hot_memories[duplicate_id]
                elif duplicate_id in self.memory_module.sleeping_memories:
                    original_memory = self.memory_module.sleeping_memories[duplicate_id]
                else:
                    # 从数据库加载
                    try:
                        self.memory_module.cursor.execute(
                            f"SELECT * FROM {self.memory_module.config.MEMORY_TABLE} WHERE id = ?",
                            (duplicate_id,)
                        )
                        
                        row = self.memory_module.cursor.fetchone()
                        if row:
                            # from ..models import MemoryItem
                            # from ..utils import blob_to_vector
                            original_memory = MemoryItem(
                                id=row['id'],
                                vector=blob_to_vector(row['vector']),
                                user_input=row['user_input'],
                                ai_response=row['ai_response'],
                                summary=row['summary'] or "",
                                heat=row['heat'],
                                created_turn=row['created_turn'],
                                last_interaction_turn=row['last_interaction_turn'],
                                access_count=row['access_count'],
                                is_hot=bool(row['is_hot']),
                                is_sleeping=bool(row['is_sleeping']),
                                cluster_id=row['cluster_id'],
                                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                                version=row['version'],
                                update_count=row['update_count'] or 0,
                                parent_turn=row['parent_turn']
                            )
                    except Exception as e:
                        print(f"[Memory] 从数据库加载重复记忆失败: {e}")
                
                if original_memory:
                    # 获取当前事实的 parent_turn
                    parent_turn = metadata.get("parent_turn")
                    if parent_turn is None and "turn" in metadata:
                        parent_turn = metadata["turn"]
                    if parent_turn is None:
                        parent_turn = metadata.get("created_turn", current_turn)
                    
                    # 初始化或更新 metadata 中的 parent_turns 列表
                    if original_memory.metadata is None:
                        original_memory.metadata = {}
                    
                    # 获取现有的 parent_turns 列表
                    parent_turns = original_memory.metadata.get("parent_turns", [])
                    
                    # 将原记忆原有的 parent_turn 加入列表（如果存在且有效）
                    if original_memory.parent_turn is not None and original_memory.parent_turn not in parent_turns:
                        parent_turns.append(original_memory.parent_turn)
                    
                    # 添加新的 parent_turn
                    if parent_turn not in parent_turns:
                        parent_turns.append(parent_turn)
                    
                    # 更新 metadata
                    original_memory.metadata["parent_turns"] = parent_turns
                    
                    # 计算新的 parent_turn 值（取最小轮次作为主要关联）
                    new_parent_turn = min(parent_turns) if parent_turns else None
                    
                    # 更新数据库
                    self.memory_module.cursor.execute(f"""
                        UPDATE {self.memory_module.config.MEMORY_TABLE}
                        SET metadata = ?, parent_turn = ?, last_interaction_turn = ?, 
                            access_count = access_count + 1, update_count = update_count + 1
                        WHERE id = ?
                    """, (
                        json.dumps(original_memory.metadata, ensure_ascii=False),
                        new_parent_turn,
                        current_turn,
                        duplicate_id
                    ))
                    
                    # 更新内存中的对象
                    if duplicate_id in self.memory_module.hot_memories:
                        mem = self.memory_module.hot_memories[duplicate_id]
                        mem.metadata = original_memory.metadata
                        mem.parent_turn = new_parent_turn
                        mem.last_interaction_turn = current_turn
                        mem.access_count += 1
                        mem.update_count += 1
                    elif duplicate_id in self.memory_module.sleeping_memories:
                        mem = self.memory_module.sleeping_memories[duplicate_id]
                        mem.metadata = original_memory.metadata
                        mem.parent_turn = new_parent_turn
                        mem.last_interaction_turn = current_turn
                        mem.access_count += 1
                        mem.update_count += 1
                    
                    # 如果原记忆在休眠区，检查是否需要唤醒
                    if duplicate_id in self.memory_module.sleeping_memories and mem.heat > 0:
                        # 通过热力系统唤醒（如果有热力）
                        self.memory_module.heat_system._unified_update_heat(
                            memory_id=duplicate_id,
                            new_heat=mem.heat,
                            old_heat=0,
                            cluster_id=mem.cluster_id,
                            update_memory=True,
                            update_cluster=True,
                            update_pool=False,
                            tx=self
                        )
                        # 移入热区
                        self.memory_module.hot_memories[duplicate_id] = mem
                        del self.memory_module.sleeping_memories[duplicate_id]
                        self.memory_module.stats['hot_memories'] += 1
                        self.memory_module.stats['cold_memories'] -= 1
                    
                    self.memory_module.stats['duplicate_skipped_count'] += 1
                    memory_ids.append(duplicate_id)  # 返回已有记忆的 ID
                    print(f"[Memory] 原子事实已合并到记忆 {duplicate_id[:8]}...，现在关联 {len(parent_turns)} 个原始对话")
                    continue  # 跳过创建新记忆
                # 如果原记忆加载失败，则回退创建新记忆（很少发生）
            
            # ========== 原有逻辑：创建新记忆 ==========
            fact_id = hashlib.md5(f"{fact}_{current_turn}_{i}_{time.time()}".encode()).hexdigest()[:16]
            
            memory = MemoryItem(
                id=fact_id,
                vector=fact_vector,
                user_input=user_input,
                ai_response=ai_response,
                summary=fact[:100],
                heat=0,
                created_turn=current_turn,
                last_interaction_turn=current_turn,
                access_count=0,
                is_hot=False,
                is_sleeping=False,
                metadata={
                    **metadata,
                    "atomic_fact": fact,
                    "fact_index": i,
                    "parent_turn": metadata.get("turn", current_turn),
                    "parent_turns": [metadata.get("turn", current_turn)],  # 初始化 parent_turns 列表
                    "type": "atomic_fact",
                    "source": "atomic_extraction"
                },
                parent_turn=metadata.get("turn")
            )
            
            # 查找最适合的簇
            best_cluster_id, best_similarity = self.memory_module._find_best_cluster_annoy(fact_vector)
            
            if best_cluster_id and best_similarity >= self.memory_module.config.CLUSTER_SIMILARITY_THRESHOLD:
                cluster = self.memory_module.clusters.get(best_cluster_id)
                if cluster:
                    with cluster.lock:
                        cluster.memory_ids.add(fact_id)
                        cluster.size += 1
                        cluster.cold_memory_count += 1
                    memory.cluster_id = best_cluster_id
                    self.memory_module.memory_to_cluster[fact_id] = best_cluster_id
            else:
                # 创建新簇
                cluster_id = f"atomic_cluster_{current_turn}_{hashlib.md5(fact_vector.tobytes()).hexdigest()[:8]}"
                cluster = SemanticCluster(
                    id=cluster_id,
                    centroid=fact_vector.copy(),
                    total_heat=0,
                    hot_memory_count=0,
                    cold_memory_count=1,
                    is_loaded=True,
                    size=1,
                    last_updated_turn=current_turn,
                    memory_additions_since_last_update=0
                )
                self.memory_module.clusters[cluster_id] = cluster
                self.memory_module.cluster_vectors[cluster_id] = cluster.centroid
                self.memory_module._update_cluster_index(cluster_id, fact_vector, 'add')
                self.memory_module.stats['clusters'] = self.memory_module.stats.get('clusters', 0) + 1
                self.memory_module._save_cluster_to_db(cluster)
                memory.cluster_id = cluster_id
                self.memory_module.memory_to_cluster[fact_id] = cluster_id
            
            # 加入休眠区
            self.memory_module.sleeping_memories[fact_id] = memory
            
            # 保存到数据库
            try:
                self.memory_module.cursor.execute(f"""
                    INSERT INTO {self.memory_module.config.MEMORY_TABLE} 
                    (id, vector, user_input, ai_response, summary, heat, created_turn, last_interaction_turn, 
                     access_count, is_hot, is_sleeping, cluster_id, metadata, update_count, parent_turn)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """, (
                    memory.id,
                    vector_to_blob(memory.vector),
                    memory.user_input,
                    memory.ai_response,
                    memory.summary,
                    memory.heat,
                    memory.created_turn,
                    memory.last_interaction_turn,
                    memory.access_count,
                    0,
                    1,  # is_sleeping = 1
                    memory.cluster_id,
                    json.dumps(memory.metadata, ensure_ascii=False),
                    memory.parent_turn
                ))
            except Exception as e:
                print(f"[Memory] 保存原子事实到数据库失败: {e}")
                continue
            
            memory_ids.append(fact_id)
            self.memory_module.stats['cold_memories'] = self.memory_module.stats.get('cold_memories', 0) + 1
            self.memory_module.stats['total_memories'] = self.memory_module.stats.get('total_memories', 0) + 1
            print(f"[Memory] 创建新原子记忆 {fact_id[:8]}...，关联原始对话轮次 {memory.parent_turn}")
        
        if memory_ids:
            print(f"[Memory] 成功存储 {len(memory_ids)} 条原子事实")
        
        self.memory_module._trigger_maintenance_if_needed()
        return memory_ids


class MemoryModule:
    def __init__(self, embedding_func=None, similarity_func=None):
        self.config = Config()

        self.embedding_dim = self.config.EMBEDDING_DIM

        self.current_turn = self.config.INITIAL_TURN
        self.turn_lock = threading.RLock()

        self.CHECKPOINT_MEMORY_THRESHOLD = 100
        self.CONSISTENCY_CHECK_THRESHOLD = 50
        self.MAINTENANCE_OPERATION_THRESHOLD = 200

        self.memory_addition_count = 0
        self.operation_count = 0
        self.memory_additions_since_last_centroid_update: int = 0
        self.maintenance_cycles_since_heat_check: int = 0

        self._external_embedding_func = embedding_func
        self._external_similarity_func = similarity_func

        if self._external_embedding_func is None:
            self._init_model()
        else:
            self.model = None

        # 数据库
        self.db = Database(self.config)
        self.conn = self.db.conn
        self.cursor = self.db.cursor

        # 基础设施
        self.lock_manager = DistributedLockManager()
        self.cache_manager = CacheManager(self.config)

        # 核心状态
        self.hot_memories: Dict[str, MemoryItem] = {}
        self.sleeping_memories: Dict[str, MemoryItem] = {}
        self.clusters: Dict[str, SemanticCluster] = {}
        self.cluster_vectors: Dict[str, np.ndarray] = {}
        self.memory_to_cluster: Dict[str, str] = {}

        # 热力系统
        self.heat_pool: int = 0
        self.unallocated_heat: int = 0
        self.total_allocated_heat: int = 0
        self.heat_pool_lock = threading.RLock()

        self.clusters_needing_centroid_update: Set[str] = set()
        self.duplicate_skipped_count = 0

        self.update_queue = queue.Queue()
        self.operation_log: deque = deque(maxlen=10000)

        self.background_executor = ThreadPoolExecutor(max_workers=self.config.BACKGROUND_THREADS)
        self.running = True

        self.access_frequency_stats: Dict[str, Dict[str, Any]] = {}
        self.frequency_stats_lock = threading.RLock()

        self.last_heat_recycle_turn: int = 0
        self.heat_recycle_count: int = 0
        self.cluster_heat_history: Dict[str, List[Tuple[int, int]]] = {}

        self.stats = {
            'total_memories': 0,
            'hot_memories': 0,
            'cold_memories': 0,
            'clusters': 0,
            'loaded_clusters': 0,
            'total_heat_recycled': 0,
            'total_heat_allocated': 0,
            'last_recycle_turn': 0,
            'consistency_violations': 0,
            'transaction_retries': 0,
            'centroid_updates': 0,
            'full_centroid_recalculations': 0,
            'maintenance_cycles': 0,
            'events_triggered': 0,
            'cluster_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'weight_adjustments': 0,
            'high_frequency_memories': 0,
            'current_turn': self.current_turn,
            'heat_redistributions': 0,
            'heat_recycled_to_pool': 0,
            'suppressed_memory_additions': 0,
            'layered_searches': 0,
            'vectorized_searches': 0,
            'similarity_cache_hits': 0,
            'similarity_cache_misses': 0,
            'annoy_queries': 0,
            'annoy_fallback_searches': 0,
            'history_records': 0,
            'history_file_records': 0,
            'pending_heat_records': 0
        }

        # 服务层
        self.heat_system = HeatSystem(self)
        self.cluster_service = ClusterService(self)
        self.search_service = SearchService(self)
        # === Waypoint Integration: initialize waypoint service ===
        self.waypoint_service = WaypointService(self)

        # 历史记录管理器
        self.history_manager = HistoryManager(
            memory_module=self,
            history_file=self.config.HISTORY_FILE_PATH,
            max_memory_records=self.config.HISTORY_MAX_MEMORY_RECORDS,
            max_disk_records=self.config.HISTORY_MAX_DISK_RECORDS
        )

        self._load_system_state()

        if self.cluster_service.cluster_index and len(self.clusters) > 0:
            self.cluster_service._rebuild_cluster_index()

        print(f"Memory Module initialized")

    def add_atomic_memories(self, facts: List[str], user_input: str, ai_response: str,
                            metadata: Dict[str, Any] = None) -> List[str]:
        """将原子事实存储为独立的记忆项"""
        with TransactionContext(self, consistency_level=ConsistencyLevel.STRONG) as tx:
            return tx.add_atomic_memories(facts, user_input, ai_response, metadata)

    def _init_model(self):
        """初始化嵌入模型"""
        if self._external_embedding_func is not None:
            print(f"Using external embedding function")
            self.embedding_dim = self.config.EMBEDDING_DIM
            return
        try:
            from sentence_transformers import SentenceTransformer
            if self.config.MODEL_PATH:
                self.model = SentenceTransformer(self.config.MODEL_PATH)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"Model loaded from {self.config.MODEL_PATH}, dimension: {self.embedding_dim}")
            else:
                warnings.warn("No model path configured. Using random embeddings.")
                self.model = None
                self.embedding_dim = self.config.EMBEDDING_DIM
        except ImportError:
            warnings.warn("sentence-transformers not installed. Using random embeddings.")
            self.model = None
            self.embedding_dim = self.config.EMBEDDING_DIM
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}. Using random embeddings.")
            self.model = None
            self.embedding_dim = self.config.EMBEDDING_DIM

    def search_original_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                                max_results: int = 10) -> List[Tuple[MemoryItem, float]]:
        """搜索原子事实并聚合返回原始对话记忆"""
        return self.search_service.search_original_memories(query_text, query_vector, max_results)

    def _add_memory_internal(self, user_input: str, ai_response: str, metadata: Dict[str, Any] = None) -> str:
        current_turn = self._unified_turn_increment('add')
        user_vector = self._get_embedding(user_input)
        summary = self._generate_summary(user_input, ai_response)

        duplicate_id = self._check_duplicate(user_vector, user_input)
        if duplicate_id:
            if duplicate_id in self.hot_memories:
                memory = self.hot_memories[duplicate_id]
                memory.last_interaction_turn = current_turn
                memory.access_count += 1
                # 使用事务上下文更新热力
                with TransactionContext(self, consistency_level=ConsistencyLevel.STRONG) as tx:
                    self.heat_system._unified_update_heat(
                        duplicate_id, memory.heat,
                        update_memory=False, update_cluster=False, update_pool=False,
                        tx=tx
                    )
                    self.cursor.execute(f"""
                        UPDATE {self.config.MEMORY_TABLE}
                        SET last_interaction_turn = ?, access_count = ?, 
                            update_count = update_count + 1
                        WHERE id = ?
                    """, (current_turn, memory.access_count, duplicate_id))
            self.duplicate_skipped_count += 1
            self._trigger_maintenance_if_needed()
            return duplicate_id

        memory_id = hashlib.md5(f"{user_input}_{ai_response}_{current_turn}".encode()).hexdigest()[:16]

        # 热力池检查 - 通过 HeatSystem 处理
        pool_threshold = self.config.INITIAL_HEAT_POOL * self.config.HEAT_POOL_RECYCLE_THRESHOLD
        if self.heat_pool < pool_threshold:
            self.heat_system._recycle_heat_pool()
        if self.heat_pool < self.config.NEW_MEMORY_HEAT:
            need_heat = self.config.NEW_MEMORY_HEAT - self.heat_pool
            self.heat_system._recycle_from_memories(need_heat)

        # 创建新记忆对象
        memory = MemoryItem(
            id=memory_id,
            vector=user_vector,
            user_input=user_input,
            ai_response=ai_response,
            summary=summary,
            heat=0,
            created_turn=current_turn,
            last_interaction_turn=current_turn,
            metadata=metadata or {}
        )

        # 基础可分配热力
        base_allocated_heat = min(self.config.NEW_MEMORY_HEAT, self.heat_pool)
        suppression_factor = self.heat_system._get_suppression_factor()
        if suppression_factor < 1.0:
            allocated_heat = int(base_allocated_heat * suppression_factor)
        else:
            allocated_heat = base_allocated_heat

        # 首先找到最适合的簇
        best_cluster_id, best_similarity = self._find_best_cluster_annoy(user_vector)

        # 使用事务包裹整个添加过程
        with TransactionContext(self, consistency_level=ConsistencyLevel.STRONG) as tx:
            # 标记：是否需要跳过邻居分配
            skip_neighbor_allocation = False
            new_memory_final_heat = allocated_heat

            # 冷主导簇处理
            if best_cluster_id and best_similarity >= self.config.CLUSTER_SIMILARITY_THRESHOLD:
                new_memory_final_heat, skip_neighbor_allocation = self.heat_system.allocate_heat_with_cold_dominant(
                    best_cluster_id, allocated_heat, best_similarity, tx
                )

            # 先从热力池扣除 allocated_heat
            tx.add_pool_update(-allocated_heat)

            # 邻居热力分配
            if not skip_neighbor_allocation:
                neighbors = self._find_neighbors(user_vector, exclude_id=memory_id)
                if neighbors:
                    neighbor_count = min(len(neighbors), 5)
                    total_neighbor_heat = allocated_heat // 2
                    heat_per_neighbor = total_neighbor_heat // neighbor_count if neighbor_count > 0 else 0
                    new_memory_final_heat = allocated_heat - total_neighbor_heat

                    # 分配热力给邻居
                    for (neighbor_id, _, neighbor_memory) in neighbors[:neighbor_count]:
                        new_neighbor_heat = neighbor_memory.heat + heat_per_neighbor
                        neighbor_cluster_id = neighbor_memory.cluster_id
                        tx.add_memory_heat_update(
                            memory_id=neighbor_id,
                            old_heat=neighbor_memory.heat,
                            new_heat=new_neighbor_heat,
                            cluster_id=neighbor_cluster_id
                        )
                    memory.heat = new_memory_final_heat
                else:
                    # 没有邻居，新记忆获得全部热力
                    memory.heat = allocated_heat
                    new_memory_final_heat = allocated_heat
            else:
                # 冷主导簇：已经设置了 new_memory_final_heat，直接给新记忆
                memory.heat = new_memory_final_heat

            # 确保新记忆的热力被正确设置
            if memory.heat > 0:
                tx.add_memory_heat_update(
                    memory_id=memory_id,
                    old_heat=0,
                    new_heat=memory.heat,
                    cluster_id=best_cluster_id
                )

            # 分配簇
            if best_cluster_id and best_similarity >= self.config.CLUSTER_SIMILARITY_THRESHOLD:
                cluster_id = best_cluster_id
                cluster = self.clusters[cluster_id]
                with cluster.lock:
                    cluster.memory_ids.add(memory_id)
                    cluster.size += 1
                    cluster.hot_memory_count += 1
                    cluster.total_heat += memory.heat
                memory.cluster_id = cluster_id
                self.memory_to_cluster[memory_id] = cluster_id

                # 更新簇热力
                tx.add_cluster_heat_update(cluster_id, memory.heat)
            else:
                # 创建新簇
                cluster_id = f"cluster_{self.current_turn}_{hashlib.md5(user_vector.tobytes()).hexdigest()[:8]}"
                cluster = SemanticCluster(
                    id=cluster_id,
                    centroid=user_vector.copy(),
                    total_heat=memory.heat,
                    hot_memory_count=1,
                    cold_memory_count=0,
                    is_loaded=True,
                    size=1,
                    last_updated_turn=self.current_turn,
                    memory_additions_since_last_update=0
                )
                self.clusters[cluster_id] = cluster
                self.cluster_vectors[cluster_id] = cluster.centroid
                self._update_cluster_index(cluster_id, user_vector, 'add')
                self.stats['clusters'] += 1
                self._save_cluster_to_db(cluster)
                memory.cluster_id = cluster_id
                self.memory_to_cluster[memory_id] = cluster_id

            # 将记忆加入热区
            self.hot_memories[memory_id] = memory

            self._update_access_frequency(memory_id)

            # 保存到数据库
            self.cursor.execute(f"""
                INSERT INTO {self.config.MEMORY_TABLE} 
                (id, vector, user_input, ai_response, summary, heat, created_turn, last_interaction_turn, 
                 access_count, is_hot, is_sleeping, cluster_id, metadata, update_count, parent_turn)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL)
            """, (
                memory.id,
                vector_to_blob(memory.vector),
                memory.user_input,
                memory.ai_response,
                memory.summary,
                memory.heat,
                memory.created_turn,
                memory.last_interaction_turn,
                memory.access_count,
                int(memory.is_hot),
                int(memory.is_sleeping),
                memory.cluster_id,
                json.dumps(memory.metadata),
            ))

            # 添加到历史记录
            self.history_manager.add_history_record(
                created_turn=current_turn,
                memory_id=memory_id,
                content_preview=summary
            )

            # === Waypoint Integration: 为新记忆建立语义边 ===
            # 选择候选记忆（从热区中随机挑选或取热度最高的 N 个）
            candidate_limit = 50  # 可配置
            # 排除自身，取热区中热度最高的候选记忆（也可以随机采样）
            hot_items = list(self.hot_memories.items())
            hot_items.sort(key=lambda x: x[1].heat, reverse=True)
            candidate_memories = [mem for mem_id, mem in hot_items if mem_id != memory_id][:candidate_limit]
            if candidate_memories:
                self.waypoint_service.add_semantic_edges_for_new_memory(memory, candidate_memories)

            # 事务提交时自动应用所有更新

        self.stats['hot_memories'] += 1
        self.stats['total_memories'] += 1
        self.memory_addition_count += 1
        self._trigger_maintenance_if_needed()

        pending_heat = self.heat_system.pending_heat_per_cluster.get(best_cluster_id, 0) if best_cluster_id else 0
        print(f"[Memory] Added memory {memory_id} with {memory.heat:,} heat "
              f"(Pool: {self.heat_pool:,}, Unallocated: {self.unallocated_heat:,})"
              f"{' [暂存热力: ' + str(pending_heat) + ']' if skip_neighbor_allocation else ''}")

        return memory_id

    def _get_embedding(self, text: str) -> np.ndarray:
        if self._external_embedding_func is not None:
            return self._external_embedding_func(text)
        elif self.model:
            try:
                return self.model.encode(text, show_progress_bar=False)
            except:
                return self.model.encode(text)
        else:
            return np.random.randn(self.embedding_dim).astype(np.float32)

    def _find_best_cluster_annoy(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        """查找最适合的簇（代理到 cluster_service）"""
        return self.cluster_service._find_best_cluster_annoy(vector)

    def _update_cluster_index(self, cluster_id: str, centroid: np.ndarray = None, operation: str = 'add'):
        """更新簇索引（代理到 cluster_service）"""
        if self.cluster_service:
            self.cluster_service._update_cluster_index(cluster_id, centroid, operation)

    def _save_cluster_to_db(self, cluster: SemanticCluster):
        """保存新簇到数据库（代理到 cluster_service）"""
        if self.cluster_service:
            self.cluster_service._save_cluster_to_db(cluster)

    def _update_unallocated_heat(self):
        """更新未分配热力（代理到 heat_system）"""
        return self.heat_system._update_unallocated_heat()

    def _load_system_state(self):
        """加载系统状态"""
        row = self.db.load_heat_pool_state()
        if row:
            self.heat_pool = row['heat_pool']
            self.unallocated_heat = row['unallocated_heat']
            self.total_allocated_heat = row['total_allocated_heat']
            self.current_turn = row['current_turn']
            self.stats['current_turn'] = self.current_turn

            # 加载暂存热力状态
            try:
                self.cursor.execute(f"""
                    SELECT cluster_id, pending_heat FROM {self.config.PENDING_HEAT_TABLE}
                    WHERE pending_heat > 0
                """)
                rows = self.cursor.fetchall()
                for db_row in rows:
                    self.heat_system.pending_heat_per_cluster[db_row['cluster_id']] = db_row['pending_heat']
                self.stats['pending_heat_records'] = len(rows)
                print(f"Loaded {len(rows)} pending heat records from database")
            except Exception as e:
                print(f"Warning: Failed to load pending heat: {e}")

        clusters = self.db.load_all_clusters()
        for cluster in clusters:
            self.clusters[cluster.id] = cluster
            self.cluster_vectors[cluster.id] = cluster.centroid
        print(f"Loaded {len(self.clusters)} clusters from database")

        memories = self.db.load_hot_memories(limit=1000)
        for memory in memories:
            self.hot_memories[memory.id] = memory
            self.memory_to_cluster[memory.id] = memory.cluster_id
            if memory.cluster_id and memory.cluster_id in self.clusters:
                self.clusters[memory.cluster_id].memory_ids.add(memory.id)
                if not self.clusters[memory.cluster_id].is_loaded:
                    self.clusters[memory.cluster_id].is_loaded = True
        print(f"Loaded {len(self.hot_memories)} hot memories from database")
        self.cache_manager.rebuild_vector_cache(self.hot_memories)

        self._load_access_frequency_stats()
        self._check_consistency()

        print(f"System state loaded. Current turn: {self.current_turn}")
        print(f"Heat system: Pool={self.heat_pool:,}, Unallocated={self.unallocated_heat:,}, "
              f"Memory heat={sum(m.heat for m in self.hot_memories.values()):,}")

    def _load_access_frequency_stats(self):
        try:
            for memory_id, memory in self.hot_memories.items():
                self.access_frequency_stats[memory_id] = {
                    'count': memory.access_count,
                    'last_reset_turn': memory.created_turn,
                    'recent_interactions': []
                }
        except Exception as e:
            print(f"Warning: Failed to load access frequency stats: {e}")

    def _check_consistency(self):
        """检查系统一致性"""
        print("Checking system consistency...")
        total_heat_in_memories = 0
        total_heat_in_clusters = 0
        for memory in self.hot_memories.values():
            total_heat_in_memories += memory.heat
        self.cursor.execute(f"SELECT SUM(heat) as total FROM {self.config.MEMORY_TABLE} WHERE is_hot = 0")
        row = self.cursor.fetchone()
        cold_heat = row['total'] or 0
        total_heat_in_memories += cold_heat
        for cluster in self.clusters.values():
            total_heat_in_clusters += cluster.total_heat

        self.cursor.execute(f"SELECT SUM(pending_heat) as total FROM {self.config.PENDING_HEAT_TABLE}")
        row = self.cursor.fetchone()
        total_pending = row['total'] or 0

        expected_total_allocated = total_heat_in_memories + self.heat_pool + self.unallocated_heat + total_pending
        expected_total_heat = self.config.TOTAL_HEAT
        if abs(expected_total_allocated - expected_total_heat) > 100:
            print(f"WARNING: Heat conservation violated! "
                  f"Total allocated: {expected_total_allocated:,}, "
                  f"Expected total: {expected_total_heat:,}")
            self.stats['consistency_violations'] += 1
            self._repair_consistency()
        if abs(total_heat_in_memories - total_heat_in_clusters) > 100:
            print(f"WARNING: Heat inconsistency! "
                  f"Memories: {total_heat_in_memories:,}, "
                  f"Clusters: {total_heat_in_clusters:,}")
            self.stats['consistency_violations'] += 1
            self._repair_consistency()
        print(f"Heat consistency check: "
              f"Memories={total_heat_in_memories:,}, "
              f"Clusters={total_heat_in_clusters:,}, "
              f"Pool={self.heat_pool:,}, "
              f"Unallocated={self.unallocated_heat:,}, "
              f"Pending={total_pending:,}, "
              f"Total={total_heat_in_memories + self.heat_pool + self.unallocated_heat + total_pending:,}")
        print(f"Current turn: {self.current_turn}")

    def _repair_consistency(self):
        """修复一致性"""
        print("Attempting to repair consistency...")
        for cluster_id, cluster in self.clusters.items():
            self.cursor.execute(f"""
                SELECT 
                    SUM(heat) as total_heat,
                    COUNT(CASE WHEN is_hot = 1 THEN 1 END) as hot_count,
                    COUNT(CASE WHEN is_hot = 0 THEN 1 END) as cold_count,
                    COUNT(*) as total_count
                FROM {self.config.MEMORY_TABLE}
                WHERE cluster_id = ?
            """, (cluster_id,))
            row = self.cursor.fetchone()
            if row:
                db_total_heat = row['total_heat'] or 0
                db_hot_count = row['hot_count'] or 0
                db_cold_count = row['cold_count'] or 0
                db_total_count = row['total_count'] or 0
                cluster.total_heat = db_total_heat
                cluster.hot_memory_count = db_hot_count
                cluster.cold_memory_count = db_cold_count
                cluster.size = db_total_count
                self.cursor.execute(f"""
                    UPDATE {self.config.CLUSTER_TABLE}
                    SET total_heat = ?, hot_memory_count = ?, 
                        cold_memory_count = ?, size = ?, version = version + 1
                    WHERE id = ?
                """, (db_total_heat, db_hot_count, db_cold_count, db_total_count, cluster_id))
                print(f"  Cluster {cluster_id[:8]}...: heat={db_total_heat:,}, "
                      f"hot={db_hot_count}, cold={db_cold_count}")
        self.cursor.execute(f"SELECT SUM(heat) as total_heat FROM {self.config.MEMORY_TABLE}")
        row = self.cursor.fetchone()
        total_memory_heat = row['total_heat'] or 0

        self.cursor.execute(f"SELECT SUM(pending_heat) as total FROM {self.config.PENDING_HEAT_TABLE}")
        row = self.cursor.fetchone()
        total_pending = row['total'] or 0

        allocated_heat = total_memory_heat + self.heat_pool + total_pending
        self.unallocated_heat = max(0, self.config.TOTAL_HEAT - allocated_heat)

        if self.heat_pool > self.config.INITIAL_HEAT_POOL and self.unallocated_heat < 0:
            excess = self.heat_pool - self.config.INITIAL_HEAT_POOL
            self.heat_pool -= excess
            self.unallocated_heat += excess
        with self.heat_pool_lock:
            self.cursor.execute(f"""
                UPDATE {self.config.HEAT_POOL_TABLE}
                SET heat_pool = ?, unallocated_heat = ?
                WHERE id = 1
            """, (self.heat_pool, self.unallocated_heat))
        print(f"  Heat pool: {self.heat_pool:,}")
        print(f"  Unallocated heat: {self.unallocated_heat:,}")
        print(f"  Total memory heat: {total_memory_heat:,}")
        print(f"  Pending heat: {total_pending:,}")
        print("Consistency repair completed")

    def _unified_turn_increment(self, increment_type: str = 'access') -> int:
        increment = {
            'access': self.config.TURN_INCREMENT_ON_ACCESS,
            'add': self.config.TURN_INCREMENT_ON_ADD,
            'maintenance': self.config.TURN_INCREMENT_ON_MAINTENANCE
        }.get(increment_type, 1)
        self.current_turn = increment_turn(self.current_turn, self.stats, self.turn_lock, increment)
        return self.current_turn

    def _invalidate_related_caches(self, memory_id: str = None, cluster_id: str = None, full: bool = False):
        if full:
            self.cache_manager.weight_cache.clear()
            self.cache_manager.cluster_search_cache.clear()
            self.cache_manager.similarity_cache.cache.clear()
            self.cache_manager.invalidate_vector_cache()
            return
        if memory_id:
            if memory_id in self.cache_manager.weight_cache:
                del self.cache_manager.weight_cache[memory_id]
            query_keys_to_remove = []
            for query_hash in self.cache_manager.similarity_cache.cache.keys():
                if memory_id in query_hash:
                    query_keys_to_remove.append(query_hash)
            for key in query_keys_to_remove:
                if key in self.cache_manager.similarity_cache.cache:
                    del self.cache_manager.similarity_cache.cache[key]
            self.cache_manager.invalidate_vector_cache(memory_id)
        if cluster_id:
            self.cache_manager.cluster_search_cache.clear(cluster_id)
            keys_to_remove = []
            for mem_id, weights in self.cache_manager.weight_cache.items():
                if mem_id in self.hot_memories:
                    mem = self.hot_memories[mem_id]
                    if mem.cluster_id == cluster_id:
                        keys_to_remove.append(mem_id)
            for key in keys_to_remove:
                del self.cache_manager.weight_cache[key]

    def _ensure_transaction(self, transaction_id: str):
        try:
            self.cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table'")
            self.cursor.execute("BEGIN TRANSACTION")
            self.operation_log.append({
                'transaction_id': transaction_id,
                'type': 'begin',
                'turn': self.current_turn
            })
            return True
        except sqlite3.OperationalError as e:
            if "cannot start a transaction within a transaction" in str(e):
                return False
            else:
                raise

    def _finalize_transaction(self, transaction_id: str, success: bool):
        try:
            if success:
                self.conn.commit()
                self.operation_log.append({
                    'transaction_id': transaction_id,
                    'type': 'commit',
                    'turn': self.current_turn
                })
            else:
                self.conn.rollback()
                self.operation_log.append({
                    'transaction_id': transaction_id,
                    'type': 'rollback',
                    'turn': self.current_turn
                })
            return True
        except Exception as e:
            print(f"Transaction finalization error: {e}")
            return False

    def _apply_operation(self, operation: Dict, immediate: bool = False):
        op_type = operation['type']
        try:
            if op_type == OperationType.MEMORY_HEAT_UPDATE:
                self._apply_memory_heat_update(operation, immediate)
            elif op_type == OperationType.CLUSTER_HEAT_UPDATE:
                self._apply_cluster_heat_update(operation, immediate)
            elif op_type == OperationType.PENDING_HEAT_UPDATE:
                self._apply_pending_heat_update(operation, immediate)
            elif op_type == OperationType.POOL_HEAT_UPDATE:
                self._apply_pool_update(operation, immediate)
            if immediate:
                self._log_operation(operation, applied=True)
        except Exception as e:
            print(f"Failed to apply operation {operation}: {e}")
            raise

    def _apply_memory_heat_update(self, operation: Dict, immediate: bool):
        memory_id = operation['memory_id']
        new_heat = operation['new_heat']
        cluster_id = operation['cluster_id']
        with self.lock_manager.with_lock(f"memory_{memory_id}", self.config.MEMORY_LOCK_TIMEOUT):
            if memory_id in self.hot_memories:
                memory = self.hot_memories[memory_id]
                memory.heat = new_heat
                memory.version += 1
                memory.update_count += 1
            if immediate:
                update_memory_heat_in_db(self.cursor, self.config.MEMORY_TABLE, memory_id, new_heat)
            if cluster_id and cluster_id in self.clusters:
                heat_delta = new_heat - operation['old_heat']
                if heat_delta != 0:
                    self.heat_system._update_cluster_heat(cluster_id, heat_delta, immediate)
        if memory_id in self.cache_manager.weight_cache:
            del self.cache_manager.weight_cache[memory_id]

    def _apply_cluster_heat_update(self, operation: Dict, immediate: bool):
        cluster_id = operation['cluster_id']
        heat_delta = operation['heat_delta']
        with self.lock_manager.with_lock(f"cluster_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT):
            if cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                if immediate:
                    cluster.total_heat += heat_delta
                    cluster.version += 1
                    update_cluster_heat_in_db(self.cursor, self.config.CLUSTER_TABLE, cluster_id, heat_delta)
                else:
                    cluster.pending_heat_delta += heat_delta
        self.cache_manager.weight_cache.clear()

    def _apply_pending_heat_update(self, operation: Dict, immediate: bool):
        """应用暂存热力更新"""
        cluster_id = operation['cluster_id']
        delta = operation['delta']

        with self.lock_manager.with_lock(f"pending_heat_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT):
            # 更新内存
            current = self.heat_system.pending_heat_per_cluster.get(cluster_id, 0)
            new_value = current + delta
            if new_value > 0:
                self.heat_system.pending_heat_per_cluster[cluster_id] = new_value
            else:
                self.heat_system.pending_heat_per_cluster.pop(cluster_id, None)

            # 更新数据库
            if immediate and delta != 0:
                self.cursor.execute(f"""
                    INSERT INTO {self.config.PENDING_HEAT_TABLE} 
                    (cluster_id, pending_heat, version, last_updated_turn)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(cluster_id) DO UPDATE SET
                        pending_heat = pending_heat + ?,
                        version = version + 1,
                        last_updated_turn = ?
                """, (cluster_id, delta, self.current_turn, delta, self.current_turn))

                if new_value <= 0:
                    self.cursor.execute(f"""
                        DELETE FROM {self.config.PENDING_HEAT_TABLE}
                        WHERE cluster_id = ?
                    """, (cluster_id,))

                self.stats['pending_heat_records'] = len(self.heat_system.pending_heat_per_cluster)

    def _apply_pool_update(self, operation: Dict, immediate: bool):
        """应用热力池更新"""
        pool_delta = operation['pool_delta']

        with self.heat_pool_lock:
            # 更新内存中的热力池
            self.heat_pool += pool_delta
            if self.heat_pool < 0:
                self.heat_pool = 0

            # 重新计算未分配热力
            total_memory_heat = sum(m.heat for m in self.hot_memories.values()) + \
                                sum(m.heat for m in self.sleeping_memories.values())
            total_pending = sum(self.heat_system.pending_heat_per_cluster.values())
            self.unallocated_heat = max(0, self.config.TOTAL_HEAT - total_memory_heat - self.heat_pool - total_pending)

            # 更新数据库
            if immediate:
                self.cursor.execute(f"""
                    UPDATE {self.config.HEAT_POOL_TABLE}
                    SET heat_pool = ?, unallocated_heat = ?
                    WHERE id = 1
                """, (self.heat_pool, self.unallocated_heat))

        # 热力池变更可能需要清理相关缓存
        self._invalidate_related_caches(full=True)

    def _log_operation(self, operation: Dict, applied: bool = False):
        self.cursor.execute(f"""
            INSERT INTO {self.config.OPERATION_LOG_TABLE}
            (transaction_id, operation_type, memory_id, cluster_id, 
             old_value, new_value, turn, applied)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            operation.get('transaction_id'),
            operation['type'].value,
            operation.get('memory_id'),
            operation.get('cluster_id'),
            json.dumps(operation.get('old_value', {})),
            json.dumps(operation.get('new_value', {})),
            self.current_turn,
            int(applied)
        ))

    def _apply_immediate_updates(self, operations):
        """立即应用更新"""
        for operation in operations:
            try:
                self._apply_operation(operation, immediate=True)
            except Exception as e:
                print(f"Failed to apply immediate operation {operation}: {e}")
                raise

    def _queue_eventual_updates(self, operations):
        """将更新加入队列以异步处理"""
        for operation in operations:
            if operation['type'] == OperationType.MEMORY_HEAT_UPDATE:
                queue_item = {
                    'type': 'memory_heat_update',
                    'memory_id': operation['memory_id'],
                    'new_heat': operation['new_heat'],
                    'cluster_id': operation.get('cluster_id'),
                    'turn': self.current_turn
                }
                self.update_queue.put(queue_item)
            elif operation['type'] == OperationType.CLUSTER_HEAT_UPDATE:
                queue_item = {
                    'type': 'cluster_heat_update',
                    'cluster_id': operation['cluster_id'],
                    'heat_delta': operation['heat_delta'],
                    'turn': self.current_turn
                }
                self.update_queue.put(queue_item)
            elif operation['type'] == OperationType.PENDING_HEAT_UPDATE:
                queue_item = {
                    'type': 'pending_heat_update',
                    'cluster_id': operation['cluster_id'],
                    'delta': operation['delta'],
                    'turn': self.current_turn
                }
                self.update_queue.put(queue_item)
            elif operation['type'] == OperationType.POOL_HEAT_UPDATE:
                queue_item = {
                    'type': 'pool_heat_update',
                    'pool_delta': operation['pool_delta'],
                    'turn': self.current_turn
                }
                self.update_queue.put(queue_item)

    def _trigger_maintenance_if_needed(self):
        need_maintenance = False
        if self.operation_count >= self.MAINTENANCE_OPERATION_THRESHOLD:
            need_maintenance = True
            self.stats['events_triggered'] += 1
        if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
            need_maintenance = True
            self.stats['events_triggered'] += 1

        # 定期处理暂存热力
        if self.current_turn % self.config.PENDING_HEAT_PROCESS_INTERVAL == 0:
            self.background_executor.submit(self.heat_system.process_pending_cluster_heat)

        if need_maintenance:
            self._unified_turn_increment('maintenance')
            self.background_executor.submit(self._perform_maintenance_tasks)
            if self.operation_count >= self.MAINTENANCE_OPERATION_THRESHOLD:
                self.operation_count = 0
            if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
                self.memory_addition_count = 0

    def _flush_update_queue(self):
        batch = []
        try:
            while True:
                item = self.update_queue.get_nowait()
                batch.append(item)
        except queue.Empty:
            pass
        if batch:
            self._process_batch_updates(batch)

    def _process_batch_updates(self, batch: List[Dict]):
        """批量处理更新"""
        cluster_updates = defaultdict(int)
        cluster_centroid_updates = {}
        memory_updates = {}
        pending_heat_updates = defaultdict(int)
        pool_updates = 0  # 新增：热力池批量更新

        for item in batch:
            if item['type'] == 'cluster_heat_update':
                cluster_updates[item['cluster_id']] += item['heat_delta']
            elif item['type'] == 'cluster_centroid_update':
                cluster_centroid_updates[item['cluster_id']] = {
                    'centroid': item['centroid'],
                    'turn': item.get('turn', self.current_turn)
                }
            elif item['type'] == 'memory_heat_update':
                memory_updates[item['memory_id']] = item['new_heat']
            elif item['type'] == 'pending_heat_update':
                pending_heat_updates[item['cluster_id']] += item['delta']
            elif item['type'] == 'pool_heat_update':
                pool_updates += item['pool_delta']

        with self.conn:
            # 处理簇热力更新
            for cluster_id, heat_delta in cluster_updates.items():
                if heat_delta != 0:
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET total_heat = total_heat + ?, pending_heat_delta = pending_heat_delta - ?
                        WHERE id = ?
                    """, (heat_delta, heat_delta, cluster_id))

                    if cluster_id in self.clusters:
                        with self.clusters[cluster_id].lock:
                            self.clusters[cluster_id].total_heat += heat_delta
                            self.clusters[cluster_id].pending_heat_delta -= heat_delta

            # 处理簇质心更新
            if cluster_centroid_updates:
                for cluster_id, update_data in cluster_centroid_updates.items():
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET centroid = ?, last_updated_turn = ?, version = version + 1,
                            memory_additions_since_last_update = 0
                        WHERE id = ?
                    """, (
                        vector_to_blob(update_data['centroid']),
                        update_data['turn'],
                        cluster_id
                    ))

            # 处理暂存热力更新
            for cluster_id, delta in pending_heat_updates.items():
                if delta != 0:
                    self.cursor.execute(f"""
                        INSERT INTO {self.config.PENDING_HEAT_TABLE} 
                        (cluster_id, pending_heat, version, last_updated_turn)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(cluster_id) DO UPDATE SET
                            pending_heat = pending_heat + ?,
                            version = version + 1,
                            last_updated_turn = ?
                    """, (cluster_id, delta, self.current_turn, delta, self.current_turn))

                    current = self.heat_system.pending_heat_per_cluster.get(cluster_id, 0)
                    new_value = current + delta
                    if new_value > 0:
                        self.heat_system.pending_heat_per_cluster[cluster_id] = new_value
                    else:
                        self.heat_system.pending_heat_per_cluster.pop(cluster_id, None)
                        self.cursor.execute(f"""
                            DELETE FROM {self.config.PENDING_HEAT_TABLE}
                            WHERE cluster_id = ? AND pending_heat <= 0
                        """, (cluster_id,))

            # 处理热力池更新
            if pool_updates != 0:
                with self.heat_pool_lock:
                    self.heat_pool += pool_updates
                    if self.heat_pool < 0:
                        self.heat_pool = 0

                    total_memory_heat = sum(m.heat for m in self.hot_memories.values()) + \
                                        sum(m.heat for m in self.sleeping_memories.values())
                    total_pending = sum(self.heat_system.pending_heat_per_cluster.values())
                    self.unallocated_heat = max(0, self.config.TOTAL_HEAT - total_memory_heat - self.heat_pool - total_pending)

                    self.cursor.execute(f"""
                        UPDATE {self.config.HEAT_POOL_TABLE}
                        SET heat_pool = ?, unallocated_heat = ?
                        WHERE id = 1
                    """, (self.heat_pool, self.unallocated_heat))

        # 更新记忆热力（内存中）
        for memory_id, new_heat in memory_updates.items():
            if memory_id in self.hot_memories:
                self.hot_memories[memory_id].heat = new_heat

        if cluster_updates or memory_updates or pending_heat_updates or pool_updates != 0:
            self.cache_manager.weight_cache.clear()
            self.stats['pending_heat_records'] = len(self.heat_system.pending_heat_per_cluster)

    def _check_and_move_sleeping(self):
        if len(self.sleeping_memories) == 0:
            return
        for memory_id, memory in list(self.sleeping_memories.items()):
            if memory.heat > 0:
                with TransactionContext(self) as tx:
                    self.heat_system._unified_update_heat(
                        memory_id=memory_id,
                        new_heat=0,
                        old_heat=memory.heat,
                        cluster_id=memory.cluster_id,
                        update_memory=True,
                        update_cluster=True,
                        update_pool=True,
                        pool_delta=memory.heat,
                        tx=tx
                    )
            memory.is_hot = False
            memory.is_sleeping = False
            if memory.cluster_id:
                self.cluster_service._unified_centroid_management(
                    cluster_id=memory.cluster_id,
                    vector=memory.vector,
                    operation='remove',
                    memory_id=memory_id
                )
                if memory.cluster_id in self.clusters:
                    cluster = self.clusters[memory.cluster_id]
                    with cluster.lock:
                        cluster.hot_memory_count -= 1
                        cluster.cold_memory_count += 1
            self.cursor.execute(f"""
                UPDATE {self.config.MEMORY_TABLE}
                SET is_hot = 0, is_sleeping = 0, heat = 0, 
                    update_count = update_count + 1
                WHERE id = ?
            """, (memory_id,))
            del self.hot_memories[memory_id]
            del self.sleeping_memories[memory_id]
            with self.frequency_stats_lock:
                if memory_id in self.access_frequency_stats:
                    del self.access_frequency_stats[memory_id]
            self.stats['hot_memories'] -= 1
            self.stats['cold_memories'] += 1
        self._invalidate_related_caches(full=True)

    def _cleanup_access_frequency_stats(self):
        with self.frequency_stats_lock:
            memory_ids_to_remove = []
            for memory_id in self.access_frequency_stats:
                if memory_id not in self.hot_memories:
                    memory_ids_to_remove.append(memory_id)
            for memory_id in memory_ids_to_remove:
                del self.access_frequency_stats[memory_id]

    def _cleanup_cluster_heat_history(self):
        max_history_age = 1000
        clusters_to_remove = []
        for cluster_id, history in self.cluster_heat_history.items():
            fresh_history = [(turn, heat) for turn, heat in history
                           if self.current_turn - turn <= max_history_age]
            if fresh_history:
                self.cluster_heat_history[cluster_id] = fresh_history
            else:
                clusters_to_remove.append(cluster_id)
        for cluster_id in clusters_to_remove:
            del self.cluster_heat_history[cluster_id]

    def _create_checkpoint_if_needed(self):
        if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
            self._create_checkpoint()

    def _create_checkpoint(self):
        print(f"[Memory System] Creating system checkpoint (Turn: {self.current_turn})")
        try:
            for memory in self.hot_memories.values():
                update_memory_heat_in_db(self.cursor, self.config.MEMORY_TABLE, memory.id, memory.heat)
            for cluster in self.clusters.values():
                self.cursor.execute(f"""
                    UPDATE {self.config.CLUSTER_TABLE}
                    SET centroid = ?, total_heat = ?, hot_memory_count = ?, cold_memory_count = ?,
                        size = ?, last_updated_turn = ?, version = version + 1,
                        memory_additions_since_last_update = ?
                    WHERE id = ?
                """, (
                    vector_to_blob(cluster.centroid),
                    cluster.total_heat,
                    cluster.hot_memory_count,
                    cluster.cold_memory_count,
                    cluster.size,
                    self.current_turn,
                    cluster.memory_additions_since_last_update,
                    cluster.id
                ))

            # 保存暂存热力
            for cluster_id, heat in self.heat_system.pending_heat_per_cluster.items():
                if heat > 0:
                    self.cursor.execute(f"""
                        INSERT INTO {self.config.PENDING_HEAT_TABLE} 
                        (cluster_id, pending_heat, version, last_updated_turn)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(cluster_id) DO UPDATE SET
                            pending_heat = ?,
                            version = version + 1,
                            last_updated_turn = ?
                    """, (cluster_id, heat, self.current_turn, heat, self.current_turn))

            with self.heat_pool_lock:
                self.cursor.execute(f"""
                    UPDATE {self.config.HEAT_POOL_TABLE}
                    SET heat_pool = ?, unallocated_heat = ?, total_allocated_heat = ?, 
                        version = version + 1, current_turn = ?
                    WHERE id = 1
                """, (
                    self.heat_pool,
                    self.unallocated_heat,
                    self.total_allocated_heat,
                    self.current_turn
                ))
            self.conn.commit()
            self.memory_addition_count = 0
            print(f"[Memory System] Checkpoint created successfully")
        except Exception as e:
            print(f"[Memory System] Error creating checkpoint: {e}")
            self.conn.rollback()

    # =============== 核心业务方法 ===============

    def add_memory(self, user_input: str, ai_response: str, metadata: Dict[str, Any] = None) -> str:
        return self._add_memory_internal(user_input=user_input, ai_response=ai_response, metadata=metadata)

    def _generate_summary(self, user_input: str, ai_response: str) -> str:
        user_preview = user_input[:50] + ("..." if len(user_input) > 50 else "")
        ai_preview = ai_response[:50] + ("..." if len(ai_response) > 50 else "")
        return f"用户: {user_preview} | AI: {ai_preview}"

    def access_memory(self, memory_id: str) -> Optional[MemoryItem]:
        current_turn = self._unified_turn_increment('access')
        if memory_id in self.hot_memories:
            memory = self.hot_memories[memory_id]
            memory.last_interaction_turn = current_turn
            memory.access_count += 1
            self._update_access_frequency(memory_id)
            self._invalidate_related_caches(memory_id=memory_id, cluster_id=memory.cluster_id)
            self.cursor.execute(f"""
                UPDATE {self.config.MEMORY_TABLE}
                SET last_interaction_turn = ?, access_count = ?, 
                    update_count = update_count + 1
                WHERE id = ?
            """, (memory.last_interaction_turn, memory.access_count, memory_id))
            self._trigger_maintenance_if_needed()
            return memory
        return None

    def _update_access_frequency(self, memory_id: str):
        with self.frequency_stats_lock:
            if memory_id not in self.access_frequency_stats:
                self.access_frequency_stats[memory_id] = {
                    'count': 1,
                    'last_reset_turn': self.current_turn,
                    'recent_interactions': [self.current_turn]
                }
            else:
                stats = self.access_frequency_stats[memory_id]
                stats['count'] += 1
                stats['recent_interactions'].append(self.current_turn)
                if len(stats['recent_interactions']) > 100:
                    stats['recent_interactions'] = stats['recent_interactions'][-100:]
                if self.current_turn - stats['last_reset_turn'] > 1000:
                    stats['count'] = 1
                    stats['last_reset_turn'] = self.current_turn
                    stats['recent_interactions'] = [self.current_turn]
        if memory_id in self.cache_manager.weight_cache:
            del self.cache_manager.weight_cache[memory_id]

    def _check_duplicate(self, vector: np.ndarray, content: str = None) -> Optional[str]:
        if not self.config.DUPLICATE_CHECK_ENABLED:
            return None
        best_similarity = 0.0
        best_memory_id = None
        for memory_id, memory in self.hot_memories.items():
            similarity = compute_cosine_similarity(vector, memory.vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_memory_id = memory_id
            if best_similarity >= self.config.DUPLICATE_THRESHOLD:
                break
        if content and best_similarity < self.config.DUPLICATE_THRESHOLD:
            content_hash = hashlib.md5(content.strip().lower().encode()).hexdigest()
            for memory_id, memory in self.hot_memories.items():
                memory_hash = hashlib.md5(memory.user_input.strip().lower().encode()).hexdigest()
                if content_hash == memory_hash:
                    return memory_id
        if best_similarity >= self.config.DUPLICATE_THRESHOLD:
            return best_memory_id
        return None

    def _find_neighbors(self, vector: np.ndarray, exclude_id: str = None, limit: int = None) -> List[Tuple[str, float, MemoryItem]]:
        if limit is None:
            limit = self.config.MAX_NEIGHBORS
        neighbors = []
        for memory_id, memory in self.hot_memories.items():
            if exclude_id and memory_id == exclude_id:
                continue
            similarity = compute_cosine_similarity(vector, memory.vector)
            if similarity >= self.config.SIMILARITY_THRESHOLD:
                neighbors.append((memory_id, similarity, memory))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:limit]

    def enable_duplicate_detection(self, enabled: bool = True, threshold: float = None):
        self.config.DUPLICATE_CHECK_ENABLED = enabled
        if threshold is not None:
            self.config.DUPLICATE_THRESHOLD = threshold

    # =============== 历史记录方法 ===============

    def get_memory_by_turn(self, created_turn: int) -> Optional[MemoryItem]:
        return self.history_manager.get_memory_by_turn(created_turn)

    def get_turn_by_memory_id(self, memory_id: str) -> Optional[int]:
        return self.history_manager.get_turn_by_memory_id(memory_id)

    def get_memories_by_turn_range(self, start_turn: int, end_turn: int) -> List[MemoryItem]:
        return self.history_manager.get_memories_by_turn_range(start_turn, end_turn)

    def search_history_by_keyword(self, keyword: str, max_results: int = 50) -> List[Tuple[int, str, str]]:
        return self.history_manager.search_by_content_keyword(keyword, max_results)

    def get_history_stats(self) -> Dict[str, Any]:
        return self.history_manager.get_history_stats()

    def cleanup_old_history(self, max_age_turns: int = 100000):
        return self.history_manager.cleanup_old_records(max_age_turns)

    def rebuild_history_index(self):
        return self.history_manager.rebuild_index()

    # =============== 搜索方法 ===============

    def search_layered_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                               max_total_results: int = None,
                               config_override: Dict = None) -> Dict[str, LayeredSearchResult]:
        return self.search_service.search_layered_memories(query_text, query_vector, max_total_results, config_override)

    def get_layered_search_results(self, query_text: str = None, query_vector: np.ndarray = None,
                                  flatten_results: bool = True) -> List[WeightedMemoryResult]:
        return self.search_service.get_layered_search_results(query_text, query_vector, flatten_results)

    def search_within_cluster(self, query_text: str = None, query_vector: np.ndarray = None,
                             cluster_id: str = None, max_results: int = None) -> List[WeightedMemoryResult]:
        return self.search_service.search_within_cluster(query_text, query_vector, cluster_id, max_results)

    def search_similar_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                               max_results: int = 10, use_weighting: bool = True) -> List[WeightedMemoryResult]:
        return self.search_service.search_similar_memories(query_text, query_vector, max_results, use_weighting)

    def find_best_clusters_for_query(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        return self.cluster_service.find_best_clusters_for_query(query, top_k)

    def get_cluster_statistics(self, cluster_id: str) -> Dict[str, Any]:
        return self.cluster_service.get_cluster_statistics(cluster_id)

    # =============== 联想图（Waypoint）相关方法 ===============
    # === Waypoint Integration: 新增公开方法 ===
    def record_turn_memories(self, memory_ids: List[str]):
        """记录一轮对话中出现的所有记忆ID，用于建立共现边"""
        self.waypoint_service.add_cooccurrence_edges(memory_ids, self.current_turn)

    def reinforce_waypoint_edges(self, seed_ids: List[str], hit_ids: List[str]):
        """强化种子到扩散命中的边（用于搜索反馈）"""
        self.waypoint_service.reinforce_edges_from_query(seed_ids, hit_ids)

    def get_waypoint_stats(self) -> Dict[str, Any]:
        """获取联想图统计信息"""
        return self.waypoint_service.get_stats()

    # =============== 统计与清理 ===============

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache_manager.get_stats()

    def clear_all_caches(self):
        self.cache_manager.clear_all()

    def get_stats(self) -> Dict[str, Any]:
        with self.heat_pool_lock:
            stats = self.stats.copy()
            stats.update({
                'num_clusters': len(self.clusters),
                'hot_memories_count': len(self.hot_memories),
                'sleeping_memories_count': len(self.sleeping_memories),
                'heat_pool': self.heat_pool,
                'unallocated_heat': self.unallocated_heat,
                'pending_heat': sum(self.heat_system.pending_heat_per_cluster.values()),
                'pending_heat_clusters': len(self.heat_system.pending_heat_per_cluster),
                'current_turn': self.current_turn,
                # === Waypoint Integration: 添加 waypoint 统计 ===
                'waypoint_edges': self.waypoint_service.get_stats().get('total_edges', 0),
            })
            return stats

    def _perform_maintenance_tasks(self):
        """执行维护任务"""
        try:
            self._flush_update_queue()
            self._check_and_move_sleeping()
            self.cluster_service._update_cluster_centroids_batch()
            self.heat_system._check_and_adjust_heat_distribution()
            self._cleanup_access_frequency_stats()
            self._cleanup_cluster_heat_history()
            self.heat_system._audit_heat_balance()
            self._create_checkpoint_if_needed()
            # === Waypoint Integration: 可选，定期清理弱边（可在 waypoint 内部实现定时清理）===
        except Exception as e:
            print(f"[Maintenance] Error during maintenance: {e}")

    def cleanup(self):
        print(f"\n[Memory System] Cleaning up memory module (Final turn: {self.current_turn})...")

        # 保存暂存热力
        self.heat_system._save_pending_heat()

        self._perform_maintenance_tasks()

        if self.memory_addition_count > 0:
            self._create_checkpoint()

        self.background_executor.shutdown(wait=True)
        self.db.close()
        print("[Memory System] Cleanup completed")