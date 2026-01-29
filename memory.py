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

# =============== 配置常量 ===============
class Config:
    # 热力系统配置
    TOTAL_HEAT = 10000000  # 总热力值 (int32最大值)
    HEAT_POOL_RATIO = 0.1  # 热力池占总热力的比例
    INITIAL_HEAT_POOL = int(TOTAL_HEAT * HEAT_POOL_RATIO)
    
    # 新记忆分配配置
    NEW_MEMORY_HEAT = 50000  # 新记忆获得的热力
    SIMILARITY_THRESHOLD = 0.75  # 相似度阈值
    MAX_NEIGHBORS = 5  # 最大邻居数量
    
    # 重复检测配置
    DUPLICATE_THRESHOLD = 0.95  # 当相似度超过此阈值时视为重复记忆
    DUPLICATE_CHECK_ENABLED = True  # 是否启用重复检测
    
    # 热区管理配置
    HOT_ZONE_RATIO = 0.2  # 热区占比触发回收的阈值
    SINGLE_MEMORY_LIMIT_RATIO = 0.05  # 单个记忆热力上限比例
    SLEEP_MEMORY_SIZE_LIMIT = 512 * 1024 * 1024  # 休眠记忆内存限制 512MB
    
    # 冷区管理配置
    DELAYED_UPDATE_LIMIT = 10  # 延迟更新条目上限
    INITIAL_HEAT_FOR_FROZEN = 1000  # 解冻时初始热力
    
    # 语义簇配置
    CLUSTER_SIMILARITY_THRESHOLD = 0.85  # 簇内相似度阈值
    CLUSTER_MIN_SIZE = 3  # 最小簇大小
    CLUSTER_MAX_SIZE = 1000  # 最大簇大小（防止过大簇）
    CLUSTER_MERGE_THRESHOLD = 0.90  # 簇合并阈值
    CLUSTER_SPLIT_THRESHOLD = 100  # 簇分裂阈值（当簇过大时）
    
    # 质心更新配置
    CENTROID_UPDATE_FREQUENCY = 10  # 每存入N次记忆更新一次簇质心
    CENTROID_UPDATE_BATCH_SIZE = 100  # 质心更新批大小
    CENTROID_FULL_RECALC_THRESHOLD = 1000  # 当簇大小变化超过此阈值时完全重算质心
    
    # 模型配置（改为可选，由外部传入）
    MODEL_PATH = None  # 改为 None，由外部传入
    EMBEDDING_DIM = 1024  # 根据模型调整
    
    # 数据库配置
    DB_PATH = "./memory/memory.db"
    MEMORY_TABLE = "memories"
    CLUSTER_TABLE = "clusters"
    HEAT_POOL_TABLE = "heat_pool"
    STATS_TABLE = "system_stats"
    OPERATION_LOG_TABLE = "operation_log"  # 操作日志表
    
    # 系统参数
    BACKGROUND_THREADS = 4  # 后台线程数
    CLUSTER_LOAD_BATCH_SIZE = 50  # 簇加载批大小
    OPERATION_BATCH_SIZE = 100  # 批量操作大小
    
    # 事务配置
    TRANSACTION_TIMEOUT = 30  # 事务超时时间（秒）
    MAX_RETRY_COUNT = 3  # 最大重试次数
    
    # 锁配置
    MEMORY_LOCK_TIMEOUT = 5  # 内存锁超时时间（秒）
    CLUSTER_LOCK_TIMEOUT = 3  # 簇锁超时时间（秒）
    
    # 新增：基于轮数的时间系统配置
    INITIAL_TURN = 0  # 初始轮数
    TURN_INCREMENT_ON_ACCESS = 1  # 每次访问记忆时增加的轮数
    TURN_INCREMENT_ON_ADD = 1  # 每次添加记忆时增加的轮数
    TURN_INCREMENT_ON_MAINTENANCE = 1  # 每次维护时增加的轮数
    
    # 新增：簇内搜索配置
    CLUSTER_SEARCH_MAX_RESULTS = 20  # 簇内搜索返回的最大记忆数量
    ACCESS_FREQUENCY_DISCOUNT_THRESHOLD = 50  # 访问频率折扣阈值（超过此次数开始减权）
    ACCESS_FREQUENCY_DISCOUNT_FACTOR = 0.7  # 访问频率折扣因子
    RECENCY_WEIGHT_DECAY_PER_TURN = 0.001  # 每轮访问权重衰减因子
    RELATIVE_HEAT_WEIGHT_POWER = 0.5  # 相对热力权重幂指数（用于平滑权重分布）
    
    # 新增：簇内搜索缓存配置
    CLUSTER_SEARCH_CACHE_SIZE = 50  # 簇内搜索缓存大小（按簇ID）
    CLUSTER_SEARCH_CACHE_TTL_TURNS = 100  # 簇内搜索缓存TTL（轮数）
    
    # 新增：热力分布控制配置
    TOP3_HEAT_LIMIT_RATIO = 0.60  # 前3个簇热力占比上限 60%
    TOP5_HEAT_LIMIT_RATIO = 0.75  # 前5个簇热力占比上限 75%
    HEAT_RECYCLE_CHECK_FREQUENCY = 2  # 每2次维护任务检查一次热力分布
    HEAT_RECYCLE_SUPPRESSION_TURNS = 50  # 热力回收抑制期轮数
    HEAT_SUPPRESSION_FACTOR = 0.7  # 抑制期内新增热力系数
    MIN_CLUSTER_HEAT_AFTER_RECYCLE = 100  # 回收后簇的最小热力
    HEAT_RECYCLE_RATE = 0.1  # 每次回收比例（线性）
    
    # 新增：分层记忆读取配置
    LAYERED_SEARCH_CONFIG = {
        "layer_1": {
            "similarity_range": (0.75, 0.80),
            "max_results": 2,
            "heat_weight_factor": 0.3,  # 热力权重系数
            "frequency_weight_factor": 0.5,  # 频率权重系数
            "recency_weight_factor": 0.8,  # 最近访问权重系数
            "base_score_factor": 1.0,  # 基础相似度权重
            "min_heat_required": 10  # 最低热力要求
        },
        "layer_2": {
            "similarity_range": (0.80, 0.85),
            "max_results": 2,
            "heat_weight_factor": 0.5,
            "frequency_weight_factor": 0.7,
            "recency_weight_factor": 0.9,
            "base_score_factor": 1.0,
            "min_heat_required": 20
        },
        "layer_3": {
            "similarity_range": (0.85, 0.95),
            "max_results": 4,
            "heat_weight_factor": 0.8,
            "frequency_weight_factor": 0.9,
            "recency_weight_factor": 1.0,
            "base_score_factor": 1.0,
            "min_heat_required": 30
        }
    }
    
    # 分层搜索的全局配置
    LAYERED_SEARCH_ENABLED = True  # 是否启用分层搜索
    LAYERED_SEARCH_FALLBACK = False  # 当某个层找不到足够记忆时，是否用其他层补充
    LAYERED_SEARCH_MAX_TOTAL_RESULTS = 8  # 最大总返回结果数
    LAYERED_SEARCH_DEDUPLICATE = True  # 是否跨层去重

# =============== 枚举和常量 ===============
class OperationType(Enum):
    """操作类型"""
    MEMORY_HEAT_UPDATE = "memory_heat_update"
    CLUSTER_HEAT_UPDATE = "cluster_heat_update"
    MEMORY_TO_COLD = "memory_to_cold"
    MEMORY_TO_HOT = "memory_to_hot"
    CLUSTER_CREATE = "cluster_create"
    CLUSTER_UPDATE = "cluster_update"
    CLUSTER_DELETE = "cluster_delete"

class ConsistencyLevel(Enum):
    """一致性级别"""
    EVENTUAL = "eventual"  # 最终一致性
    IMMEDIATE = "immediate"  # 立即一致性（同步）
    STRONG = "strong"  # 强一致性（事务）

# =============== 事务上下文 ===============
class TransactionContext:
    """事务上下文，确保操作的原子性"""
    
    def __init__(self, memory_module, consistency_level: ConsistencyLevel = ConsistencyLevel.STRONG):
        self.memory_module = memory_module
        self.consistency_level = consistency_level
        self.operations = []  # 记录所有操作
        self.memory_updates = {}  # memory_id -> (old_heat, new_heat)
        self.cluster_updates = {}  # cluster_id -> heat_delta
        self.transaction_id = str(uuid.uuid4())
        self.transaction_started = False  # 跟踪事务是否由我们开始
    
    def __enter__(self):
        """进入事务上下文"""
        if self.consistency_level == ConsistencyLevel.STRONG:
            # 使用新方法确保不重复开始事务
            self.transaction_started = self.memory_module._ensure_transaction(self.transaction_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出事务上下文"""
        if exc_type is None:
            if self.consistency_level == ConsistencyLevel.STRONG:
                success = True
                # 应用操作
                for op in self.operations:
                    try:
                        self.memory_module._apply_operation(op, immediate=True)
                    except Exception as e:
                        print(f"Failed to apply operation: {e}")
                        success = False
                
                # 只在我们开始的事务中进行提交/回滚
                if self.transaction_started:
                    self.memory_module._finalize_transaction(self.transaction_id, success)
                    if not success:
                        raise Exception("Transaction failed")
            elif self.consistency_level == ConsistencyLevel.IMMEDIATE:
                self.memory_module._apply_immediate_updates(self.operations)
            else:  # EVENTUAL
                self.memory_module._queue_eventual_updates(self.operations)
        else:
            if self.consistency_level == ConsistencyLevel.STRONG and self.transaction_started:
                self.memory_module._finalize_transaction(self.transaction_id, False)
    
    def add_memory_heat_update(self, memory_id: str, old_heat: int, new_heat: int, cluster_id: Optional[str] = None):
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
        self.memory_updates[memory_id] = (old_heat, new_heat)
    
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

# =============== 数据类定义 ===============
@dataclass
class MemoryItem:
    """记忆项 - 使用轮数作为时间锚点"""
    id: str
    vector: np.ndarray
    content: str
    heat: int = 0
    created_turn: int = 0  # 创建时的轮数（替换created_at）
    last_interaction_turn: int = 0  # 上一次交互的轮数（替换last_accessed）
    access_count: int = 1
    is_hot: bool = True
    is_sleeping: bool = False
    cluster_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1  # 版本号，用于乐观锁
    update_count: int = 0  # 记忆更新次数（数据库维护）
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['vector'] = self.vector.tolist() if hasattr(self.vector, 'tolist') else self.vector
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        """从字典创建"""
        data = data.copy()
        if 'vector' in data and isinstance(data['vector'], list):
            data['vector'] = np.array(data['vector'], dtype=np.float32)
        return cls(**data)

@dataclass
class SemanticCluster:
    """语义簇"""
    id: str
    centroid: np.ndarray
    total_heat: int = 0
    hot_memory_count: int = 0
    cold_memory_count: int = 0
    memory_ids: Set[str] = field(default_factory=set)  # 使用集合避免重复
    is_loaded: bool = False
    size: int = 0
    last_updated_turn: int = 0  # 上次更新的轮数
    version: int = 1
    lock: threading.RLock = field(default_factory=threading.RLock)  # 每个簇有自己的锁
    pending_heat_delta: int = 0  # 待应用的热力变化
    pending_centroid_updates: List[Tuple[np.ndarray, bool]] = field(default_factory=list)  # (向量, add=True/remove=False)
    memory_additions_since_last_update: int = 0  # 上次更新后新增的记忆数

# =============== 加权记忆搜索结果 ===============
@dataclass
class WeightedMemoryResult:
    """加权记忆搜索结果"""
    memory: MemoryItem
    base_similarity: float  # 基础相似度（与查询向量的相似度）
    relative_heat_weight: float  # 相对热力权重
    access_frequency_weight: float  # 访问频率权重
    recency_weight: float  # 最近访问权重（基于轮数间隔）
    final_score: float  # 最终加权得分
    ranking_position: int  # 排名位置

# =============== 分层搜索结果 ===============
@dataclass
class LayeredSearchResult:
    """分层搜索结果"""
    layer_name: str
    similarity_range: Tuple[float, float]
    results: List[WeightedMemoryResult]
    achieved_count: int
    target_count: int
    avg_similarity: float
    avg_final_score: float

# =============== 簇内搜索缓存 ===============
class ClusterSearchCache:
    """簇内搜索缓存（基于轮数的TTL）"""
    
    def __init__(self, max_size: int = 50, ttl_turns: int = 100):
        self.max_size = max_size
        self.ttl_turns = ttl_turns
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_turns: Dict[str, int] = {}  # 使用轮数而不是时间戳
        self.lock = threading.RLock()
    
    def get(self, cluster_id: str, query_vector: np.ndarray, current_turn: int) -> Optional[List[WeightedMemoryResult]]:
        """从缓存获取搜索结果"""
        with self.lock:
            if cluster_id not in self.cache:
                return None
            
            cache_entry = self.cache[cluster_id]
            
            # 检查缓存是否过期（基于轮数）
            if current_turn - cache_entry['created_turn'] > self.ttl_turns:
                del self.cache[cluster_id]
                del self.access_turns[cluster_id]
                return None
            
            # 检查查询向量是否相同（使用近似比较）
            cached_vector = cache_entry['query_vector']
            if not self._vectors_similar(query_vector, cached_vector):
                return None
            
            # 更新访问轮数
            self.access_turns[cluster_id] = current_turn
            
            return cache_entry['results']
    
    def put(self, cluster_id: str, query_vector: np.ndarray, results: List[WeightedMemoryResult], current_turn: int):
        """将搜索结果放入缓存"""
        with self.lock:
            # 如果缓存已满，移除最久未访问的条目
            if len(self.cache) >= self.max_size:
                # 找到最久未访问的簇
                oldest_cluster = None
                oldest_turn = float('inf')
                for cid, aturn in self.access_turns.items():
                    if aturn < oldest_turn:
                        oldest_turn = aturn
                        oldest_cluster = cid
                
                if oldest_cluster:
                    del self.cache[oldest_cluster]
                    del self.access_turns[oldest_cluster]
            
            # 添加新条目
            self.cache[cluster_id] = {
                'query_vector': query_vector.copy(),
                'results': results,
                'created_turn': current_turn
            }
            self.access_turns[cluster_id] = current_turn
    
    def clear(self, cluster_id: str = None):
        """清除缓存"""
        with self.lock:
            if cluster_id:
                if cluster_id in self.cache:
                    del self.cache[cluster_id]
                if cluster_id in self.access_turns:
                    del self.access_turns[cluster_id]
            else:
                self.cache.clear()
                self.access_turns.clear()
    
    def _vectors_similar(self, vec1: np.ndarray, vec2: np.ndarray, threshold: float = 0.99) -> bool:
        """检查两个向量是否相似（用于缓存键值比较）"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return False
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return similarity >= threshold

# =============== 向量缓存系统 ===============
@dataclass
class VectorCache:
    """向量缓存系统"""
    vectors: np.ndarray = None  # (M, d) 所有向量
    memory_ids: List[str] = None  # 对应ID列表
    last_updated: float = 0
    is_valid: bool = False

class SimilarityCache:
    """相似度计算结果缓存"""
    def __init__(self, max_size=100, ttl_seconds=300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict] = {}  # query_hash -> {similarities, timestamp}
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, query_vector: np.ndarray) -> Optional[np.ndarray]:
        """获取缓存的相似度结果"""
        query_hash = self._hash_vector(query_vector)
        
        with self.lock:
            if query_hash not in self.cache:
                self.miss_count += 1
                return None
            
            cache_entry = self.cache[query_hash]
            
            # 检查是否过期
            if time.time() - cache_entry['timestamp'] > self.ttl_seconds:
                del self.cache[query_hash]
                self.miss_count += 1
                return None
            
            self.hit_count += 1
            return cache_entry['similarities']
    
    def put(self, query_vector: np.ndarray, similarities: np.ndarray):
        """存储相似度结果"""
        query_hash = self._hash_vector(query_vector)
        
        with self.lock:
            # LRU淘汰
            if len(self.cache) >= self.max_size:
                oldest_key = None
                oldest_time = float('inf')
                
                for key, entry in self.cache.items():
                    if entry['timestamp'] < oldest_time:
                        oldest_time = entry['timestamp']
                        oldest_key = key
                
                if oldest_key:
                    del self.cache[oldest_key]
            
            self.cache[query_hash] = {
                'similarities': similarities.copy(),
                'timestamp': time.time()
            }
    
    def _hash_vector(self, vector: np.ndarray) -> str:
        """向量哈希（近似）"""
        # 取前8个浮点数的字节作为哈希
        return hashlib.md5(vector[:8].tobytes()).hexdigest()[:16]
    
    def get_stats(self):
        """获取缓存统计"""
        with self.lock:
            total = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate
            }

# =============== 分布式锁管理器 ===============
class DistributedLockManager:
    """分布式锁管理器，支持超时和重试"""
    
    def __init__(self):
        self.locks: Dict[str, threading.RLock] = {}
        self.lock_timestamps: Dict[str, float] = {}
        self.lock_threads: Dict[str, int] = {}
        self.global_lock = threading.RLock()
    
    def acquire(self, lock_key: str, timeout: float = 5.0) -> bool:
        """获取锁"""
        start_time = time.time()
        thread_id = threading.get_ident()
        
        with self.global_lock:
            # 检查是否已经持有锁
            if lock_key in self.lock_threads and self.lock_threads[lock_key] == thread_id:
                return True
            
            # 尝试获取锁
            while time.time() - start_time < timeout:
                if lock_key not in self.locks:
                    self.locks[lock_key] = threading.RLock()
                    self.lock_timestamps[lock_key] = time.time()
                    self.lock_threads[lock_key] = thread_id
                    return True
                elif self.lock_timestamps[lock_key] + timeout < time.time():
                    # 锁超时，强制释放
                    del self.locks[lock_key]
                    del self.lock_timestamps[lock_key]
                    if lock_key in self.lock_threads:
                        del self.lock_threads[lock_key]
                    continue
                
                # 等待一段时间后重试
                time.sleep(0.01)
        
        return False
    
    def release(self, lock_key: str):
        """释放锁"""
        with self.global_lock:
            thread_id = threading.get_ident()
            if lock_key in self.lock_threads and self.lock_threads[lock_key] == thread_id:
                if lock_key in self.locks:
                    del self.locks[lock_key]
                    del self.lock_timestamps[lock_key]
                del self.lock_threads[lock_key]
    
    def with_lock(self, lock_key: str, timeout: float = 5.0):
        """锁上下文管理器"""
        class LockContext:
            def __init__(self, manager, lock_key, timeout):
                self.manager = manager
                self.lock_key = lock_key
                self.timeout = timeout
                self.acquired = False
            
            def __enter__(self):
                self.acquired = self.manager.acquire(self.lock_key, self.timeout)
                if not self.acquired:
                    raise TimeoutError(f"Failed to acquire lock {self.lock_key} within {self.timeout} seconds")
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.acquired:
                    self.manager.release(self.lock_key)
        
        return LockContext(self, lock_key, timeout)

# =============== 核心内存管理模块 ===============
class MemoryModule:
    """内存管理模块 - 纯事件驱动设计，基于轮数的时间系统，带NumPy向量化优化"""
    
    def __init__(self, embedding_func=None, similarity_func=None):
        self.config = Config()
        
        self.embedding_dim = self.config.EMBEDDING_DIM
        
        # 基于轮数的时间系统
        self.current_turn = self.config.INITIAL_TURN  # 当前轮数（全局时间锚点）
        self.turn_lock = threading.RLock()
        
        # 纯事件驱动配置（完全移除定时器）
        self.CHECKPOINT_MEMORY_THRESHOLD = 100  # 每添加100次记忆创建检查点
        self.CONSISTENCY_CHECK_THRESHOLD = 50   # 每50次操作检查一致性
        self.MAINTENANCE_OPERATION_THRESHOLD = 200  # 每200次操作执行完整维护
        
        # 事件计数器
        self.memory_addition_count = 0
        self.operation_count = 0
        self.memory_additions_since_last_centroid_update: int = 0
        self.maintenance_cycles_since_heat_check: int = 0  # 新增：热力分布检查计数器
        
        # 从外部传入的嵌入函数和相似度函数
        self._external_embedding_func = embedding_func
        self._external_similarity_func = similarity_func
        
        # 如果外部没有传入，使用内部默认实现
        if self._external_embedding_func is None:
            self._init_model()
        else:
            self.model = None  # 不使用内部模型
        
        # 初始化数据库和表结构
        self._init_database()
        
        # 锁管理器
        self.lock_manager = DistributedLockManager()
        
        # 内存数据结构
        self.hot_memories: Dict[str, MemoryItem] = {}  # 热区记忆
        self.sleeping_memories: Dict[str, MemoryItem] = {}  # 休眠记忆
        self.clusters: Dict[str, SemanticCluster] = {}  # 所有语义簇
        self.cluster_vectors: Dict[str, np.ndarray] = {}  # 簇质心向量缓存
        
        # 索引结构
        self.memory_to_cluster: Dict[str, str] = {}  # 记忆ID到簇ID的映射
        
        # 热力系统
        self.heat_pool: int = 0
        self.total_allocated_heat: int = 0
        self.heat_pool_lock = threading.RLock()
        
        # 统计和计数器
        self.clusters_needing_centroid_update: Set[str] = set()  # 需要更新质心的簇
        self.duplicate_skipped_count = 0  # 重复记忆跳过计数
        
        # 延迟更新队列
        self.update_queue = queue.Queue()
        self.operation_log: deque = deque(maxlen=10000)  # 操作日志
        
        # 后台线程池
        self.background_executor = ThreadPoolExecutor(max_workers=self.config.BACKGROUND_THREADS)
        
        # 纯事件驱动：没有定时维护线程
        self.running = True
        
        # 新增：簇内搜索缓存
        self.cluster_search_cache = ClusterSearchCache(
            max_size=self.config.CLUSTER_SEARCH_CACHE_SIZE,
            ttl_turns=self.config.CLUSTER_SEARCH_CACHE_TTL_TURNS
        )
        
        # 新增：访问频率统计（基于轮数）
        self.access_frequency_stats: Dict[str, Dict[str, Any]] = {}  # memory_id -> {'count': int, 'last_reset_turn': int}
        self.frequency_stats_lock = threading.RLock()
        
        # 新增：热力分布控制相关
        self.last_heat_recycle_turn: int = 0  # 上次热力回收的轮数
        self.heat_recycle_count: int = 0  # 热力回收次数统计
        self.cluster_heat_history: Dict[str, List[Tuple[int, int]]] = {}  # 簇热力历史记录 (turn, heat)
        
        # 新增：向量化优化相关缓存
        self.vector_cache = VectorCache()
        self.vector_cache_lock = threading.RLock()
        self.similarity_cache = SimilarityCache(max_size=100, ttl_seconds=300)
        self.weight_cache: Dict[str, Dict] = {}  # memory_id -> 预计算权重
        self.weight_cache_turn = 0  # 权重缓存对应的轮数
        
        # 预计算归一化向量缓存
        self._normalized_vectors: Optional[np.ndarray] = None
        self._precomputed_memory_norms: Optional[np.ndarray] = None
        self._precomputed_query_norms: Dict[str, float] = {}  # 查询向量范数缓存
        
        # 统计信息
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
        }
        
        # 加载系统状态
        self._load_system_state()
        
        print(f"Memory Module initialized with PURE event-driven design and TURN-based time system")
        print(f"NumPy vectorization and caching ENABLED")
        print(f"No background maintenance threads - all maintenance triggered by events")
        print(f"Current turn: {self.current_turn}")
        print(f"Embedding model: {'External' if self._external_embedding_func else 'Internal'}")
        print(f"Checkpoint threshold: {self.CHECKPOINT_MEMORY_THRESHOLD} memory additions")
        print(f"Consistency threshold: {self.CONSISTENCY_CHECK_THRESHOLD} operations")
        print(f"Maintenance threshold: {self.MAINTENANCE_OPERATION_THRESHOLD} total operations")
        print(f"Duplicate detection: {'Enabled' if self.config.DUPLICATE_CHECK_ENABLED else 'Disabled'} "
              f"(threshold: {self.config.DUPLICATE_THRESHOLD})")
        print(f"Cluster search enabled with cache size {self.config.CLUSTER_SEARCH_CACHE_SIZE}")
        print(f"Heat distribution control enabled: Top 3 ≤ {self.config.TOP3_HEAT_LIMIT_RATIO:.0%}, "
              f"Top 5 ≤ {self.config.TOP5_HEAT_LIMIT_RATIO:.0%}")
        print(f"Layered search enabled: {'Yes' if self.config.LAYERED_SEARCH_ENABLED else 'No'}")
        print(f"Vector cache: Initialized (size: {len(self.hot_memories)})")
    
    def _increment_turn(self, increment: int = 1) -> int:
        """增加轮数并返回新的轮数"""
        with self.turn_lock:
            self.current_turn += increment
            self.stats['current_turn'] = self.current_turn
            return self.current_turn
    
    def get_current_turn(self) -> int:
        """获取当前轮数"""
        with self.turn_lock:
            return self.current_turn
    
    def _init_model(self):
        """初始化嵌入模型（只有当外部没有传入模型时才调用）"""
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
            warnings.warn(f"sentence-transformers not installed. Using random embeddings.")
            self.model = None
            self.embedding_dim = self.config.EMBEDDING_DIM
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}. Using random embeddings.")
            self.model = None
            self.embedding_dim = self.config.EMBEDDING_DIM
    
    def _init_database(self):
        """初始化数据库表结构（使用轮数而非时间戳）"""
        self.conn = sqlite3.connect(self.config.DB_PATH, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式提高并发
        self.conn.execute("PRAGMA synchronous=NORMAL")  # 平衡性能和数据安全
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # 创建统一记忆表（使用轮数而非时间戳）
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.MEMORY_TABLE} (
                id TEXT PRIMARY KEY,
                vector BLOB,
                content TEXT NOT NULL,
                heat INTEGER DEFAULT 0,
                created_turn INTEGER DEFAULT 0,
                last_interaction_turn INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 1,
                is_hot INTEGER DEFAULT 1,
                is_sleeping INTEGER DEFAULT 0,
                cluster_id TEXT,
                metadata TEXT,
                version INTEGER DEFAULT 1,
                update_count INTEGER DEFAULT 0  -- 记忆更新次数，用于乐观锁和时间锚点
            )
        """)
        
        # 创建簇表（使用轮数而非时间戳）
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.CLUSTER_TABLE} (
                id TEXT PRIMARY KEY,
                centroid BLOB,
                total_heat INTEGER DEFAULT 0,
                hot_memory_count INTEGER DEFAULT 0,
                cold_memory_count INTEGER DEFAULT 0,
                is_loaded INTEGER DEFAULT 0,
                size INTEGER DEFAULT 0,
                last_updated_turn INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                pending_heat_delta INTEGER DEFAULT 0,
                memory_additions_since_last_update INTEGER DEFAULT 0
            )
        """)
        
        # 创建热力池表
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.HEAT_POOL_TABLE} (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                heat_pool INTEGER DEFAULT {self.config.INITIAL_HEAT_POOL},
                total_allocated_heat INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                current_turn INTEGER DEFAULT {self.config.INITIAL_TURN}  -- 存储当前轮数
            )
        """)
        
        # 创建操作日志表（使用轮数而非时间戳）
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.OPERATION_LOG_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                operation_type TEXT,
                memory_id TEXT,
                cluster_id TEXT,
                old_value TEXT,
                new_value TEXT,
                turn INTEGER DEFAULT 0,  -- 操作发生的轮数
                applied INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0
            )
        """)
        
        # 创建索引
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_op_log_turn ON {self.config.OPERATION_LOG_TABLE}(turn)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_op_log_applied ON {self.config.OPERATION_LOG_TABLE}(applied)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_cluster ON {self.config.MEMORY_TABLE}(cluster_id, heat)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_hot_heat ON {self.config.MEMORY_TABLE}(is_hot, heat DESC)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_turn ON {self.config.MEMORY_TABLE}(last_interaction_turn DESC)")
        
        # 初始化表
        self.cursor.execute(f"""
            INSERT OR IGNORE INTO {self.config.HEAT_POOL_TABLE} 
            (id, heat_pool, total_allocated_heat, current_turn)
            VALUES (1, {self.config.INITIAL_HEAT_POOL}, 0, {self.config.INITIAL_TURN})
        """)
        
        self.conn.commit()
    
    def _load_system_state(self):
        """加载系统状态（包括当前轮数）"""
        # 加载热力池和当前轮数
        self.cursor.execute(f"SELECT heat_pool, total_allocated_heat, current_turn FROM {self.config.HEAT_POOL_TABLE} WHERE id = 1")
        row = self.cursor.fetchone()
        if row:
            self.heat_pool = row['heat_pool']
            self.total_allocated_heat = row['total_allocated_heat']
            self.current_turn = row['current_turn']
            self.stats['current_turn'] = self.current_turn
        
        # 加载所有簇
        self._load_all_clusters()
        
        # 加载热区记忆
        self._load_hot_memories()
        
        # 加载访问频率统计
        self._load_access_frequency_stats()
        
        # 初始一致性检查
        self._check_consistency()
        
        print(f"System state loaded. Current turn: {self.current_turn}")
    
    def _load_all_clusters(self):
        """加载所有语义簇"""
        self.cursor.execute(f"SELECT * FROM {self.config.CLUSTER_TABLE}")
        rows = self.cursor.fetchall()
        
        for row in rows:
            cluster = SemanticCluster(
                id=row['id'],
                centroid=self._blob_to_vector(row['centroid']),
                total_heat=row['total_heat'],
                hot_memory_count=row['hot_memory_count'],
                cold_memory_count=row['cold_memory_count'],
                is_loaded=bool(row['is_loaded']),
                size=row['size'],
                last_updated_turn=row['last_updated_turn'],
                version=row['version'],
                pending_heat_delta=row['pending_heat_delta'],
                memory_additions_since_last_update=row['memory_additions_since_last_update'] or 0
            )
            self.clusters[cluster.id] = cluster
            self.cluster_vectors[cluster.id] = cluster.centroid
        
        print(f"Loaded {len(self.clusters)} clusters from database")
    
    def _load_hot_memories(self):
        """加载热区记忆"""
        self.cursor.execute(f"""
            SELECT * FROM {self.config.MEMORY_TABLE} 
            WHERE is_hot = 1
            ORDER BY heat DESC
            LIMIT 1000
        """)
        
        rows = self.cursor.fetchall()
        for row in rows:
            memory = MemoryItem(
                id=row['id'],
                vector=self._blob_to_vector(row['vector']),
                content=row['content'],
                heat=row['heat'],
                created_turn=row['created_turn'],
                last_interaction_turn=row['last_interaction_turn'],
                access_count=row['access_count'],
                is_hot=bool(row['is_hot']),
                is_sleeping=bool(row['is_sleeping']),
                cluster_id=row['cluster_id'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                version=row['version'],
                update_count=row['update_count'] or 0
            )
            
            self.hot_memories[memory.id] = memory
            self.memory_to_cluster[memory.id] = memory.cluster_id
            
            # 更新簇的加载状态
            if memory.cluster_id and memory.cluster_id in self.clusters:
                cluster = self.clusters[memory.cluster_id]
                cluster.memory_ids.add(memory.id)
                if not cluster.is_loaded:
                    cluster.is_loaded = True
        
        print(f"Loaded {len(self.hot_memories)} hot memories from database")
        
        # 初始化向量缓存
        self._rebuild_vector_cache()
    
    def _load_access_frequency_stats(self):
        """加载访问频率统计"""
        try:
            # 从热区记忆加载访问频率统计
            for memory_id, memory in self.hot_memories.items():
                self.access_frequency_stats[memory_id] = {
                    'count': memory.access_count,
                    'last_reset_turn': memory.created_turn,  # 使用创建轮数作为重置点
                    'recent_interactions': []  # 用于跟踪最近交互的轮数
                }
        except Exception as e:
            print(f"Warning: Failed to load access frequency stats: {e}")
    
    def _check_consistency(self):
        """检查系统一致性"""
        print("Checking system consistency...")
        
        # 检查热力总和是否一致
        total_heat_in_memories = 0
        total_heat_in_clusters = 0
        
        # 计算热区记忆总热力
        for memory in self.hot_memories.values():
            total_heat_in_memories += memory.heat
        
        # 计算数据库中的冷区记忆热力（应该为0）
        self.cursor.execute(f"SELECT SUM(heat) as total FROM {self.config.MEMORY_TABLE} WHERE is_hot = 0")
        row = self.cursor.fetchone()
        cold_heat = row['total'] or 0
        total_heat_in_memories += cold_heat
        
        # 计算簇总热力
        for cluster in self.clusters.values():
            total_heat_in_clusters += cluster.total_heat
        
        # 检查热力池
        expected_total_heat = self.config.TOTAL_HEAT - self.heat_pool
        
        if total_heat_in_memories != total_heat_in_clusters:
            print(f"WARNING: Heat inconsistency! Memories: {total_heat_in_memories}, Clusters: {total_heat_in_clusters}")
            self.stats['consistency_violations'] += 1
            
            # 尝试修复
            self._repair_consistency()
        
        print(f"Heat consistency check: Memories={total_heat_in_memories}, Clusters={total_heat_in_clusters}, Pool={self.heat_pool}")
        print(f"Current turn: {self.current_turn}")
    
    def _repair_consistency(self):
        """修复一致性"""
        print("Attempting to repair consistency...")
        
        # 重新计算每个簇的热力
        for cluster_id, cluster in self.clusters.items():
            # 从数据库重新计算簇热力
            self.cursor.execute(f"""
                SELECT SUM(heat) as total_heat, 
                       COUNT(CASE WHEN is_hot = 1 THEN 1 END) as hot_count,
                       COUNT(CASE WHEN is_hot = 0 THEN 1 END) as cold_count
                FROM {self.config.MEMORY_TABLE}
                WHERE cluster_id = ?
            """, (cluster_id,))
            
            row = self.cursor.fetchone()
            if row:
                db_total_heat = row['total_heat'] or 0
                db_hot_count = row['hot_count'] or 0
                db_cold_count = row['cold_count'] or 0
                
                # 更新内存中的簇状态
                cluster.total_heat = db_total_heat
                cluster.hot_memory_count = db_hot_count
                cluster.cold_memory_count = db_cold_count
                cluster.size = db_hot_count + db_cold_count
                
                # 更新数据库
                self._update_cluster_in_db(cluster)
        
        print("Consistency repair completed")
    
    def _ensure_transaction(self, transaction_id: str):
        """确保事务开始，如果已经有事务则跳过"""
        try:
            # 检查是否已经有活跃事务
            self.cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table'")
            # 如果上面没有异常，说明连接是活跃的
            # 尝试执行 BEGIN TRANSACTION，如果失败说明已经有事务
            self.cursor.execute("BEGIN TRANSACTION")
            self.operation_log.append({
                'transaction_id': transaction_id,
                'type': 'begin',
                'turn': self.current_turn
            })
            return True
        except sqlite3.OperationalError as e:
            if "cannot start a transaction within a transaction" in str(e):
                # 已经有活跃事务，不重复开始
                return False
            else:
                raise
    
    def _finalize_transaction(self, transaction_id: str, success: bool):
        """完成事务（如果有的话）"""
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
    
    def _trigger_maintenance_if_needed(self):
        """事件驱动：仅在需要时触发维护任务"""
        # 检查是否达到维护阈值
        need_maintenance = False
        
        if self.operation_count >= self.MAINTENANCE_OPERATION_THRESHOLD:
            need_maintenance = True
            self.stats['events_triggered'] += 1
        
        if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
            need_maintenance = True
            self.stats['events_triggered'] += 1
        
        if need_maintenance:
            # 增加维护轮数
            self._increment_turn(self.config.TURN_INCREMENT_ON_MAINTENANCE)
            
            # 提交维护任务到线程池
            self.background_executor.submit(self._perform_maintenance_tasks)
            
            # 重置计数器（避免重复触发）
            if self.operation_count >= self.MAINTENANCE_OPERATION_THRESHOLD:
                self.operation_count = 0
            if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
                self.memory_addition_count = 0
    
    def _perform_maintenance_tasks(self):
        """执行维护任务 - 完全事件驱动"""
        current_turn = self.current_turn
        print(f"\n[Memory System] Performing maintenance tasks (Turn: {current_turn})")
        print(f"  Operations since last: {self.operation_count}")
        print(f"  Memory additions since last: {self.memory_addition_count}")
        
        start_time = time.time()
        
        # 1. 处理所有待处理更新
        self._flush_update_queue()
        
        # 2. 应用pending的簇更新
        self._apply_pending_cluster_updates()
        
        # 3. 检查是否需要更新簇质心
        if self.memory_additions_since_last_centroid_update >= self.config.CENTROID_UPDATE_FREQUENCY:
            self._update_cluster_centroids_batch()
        
        # 4. 检查一致性
        self._check_consistency()
        
        # 5. 检查热力分布（新增）
        self._check_and_adjust_heat_distribution()
        
        # 6. 检查是否需要创建检查点
        self._create_checkpoint_if_needed()
        
        # 7. 检查并处理休眠记忆
        if len(self.sleeping_memories) > 0:
            self._check_and_move_sleeping()
        
        # 8. 更新内存缓存状态
        self._update_memory_cache_state()
        
        # 9. 清理过期的访问频率统计
        self._cleanup_access_frequency_stats()
        
        # 10. 清理簇热力历史记录
        self._cleanup_cluster_heat_history()
        
        elapsed = time.time() - start_time
        self.stats['maintenance_cycles'] += 1
        
        print(f"[Memory System] Maintenance cycle {self.stats['maintenance_cycles']} completed in {elapsed:.2f}s")
    
    def _check_and_adjust_heat_distribution(self):
        """检查并调整热力分布，确保前3/5个簇的热力占比不超过阈值"""
        # 每HEAT_RECYCLE_CHECK_FREQUENCY次维护检查一次
        self.maintenance_cycles_since_heat_check += 1
        if self.maintenance_cycles_since_heat_check < self.config.HEAT_RECYCLE_CHECK_FREQUENCY:
            return
        
        self.maintenance_cycles_since_heat_check = 0
        
        print(f"[Heat Distribution] Checking cluster heat distribution (Turn: {self.current_turn})")
        
        # 收集所有簇的热力信息
        cluster_heat_list = []
        total_cluster_heat = 0
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.total_heat > 0:
                cluster_heat_list.append({
                    'cluster_id': cluster_id,
                    'heat': cluster.total_heat,
                    'size': cluster.size
                })
                total_cluster_heat += cluster.total_heat
                
                # 记录热力历史
                if cluster_id not in self.cluster_heat_history:
                    self.cluster_heat_history[cluster_id] = []
                self.cluster_heat_history[cluster_id].append((self.current_turn, cluster.total_heat))
        
        if total_cluster_heat == 0 or len(cluster_heat_list) <= 5:
            print(f"[Heat Distribution] Not enough clusters or total heat is zero")
            return
        
        # 按热力降序排序
        cluster_heat_list.sort(key=lambda x: x['heat'], reverse=True)
        
        # 计算前3和前5个簇的热力占比
        top3_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:3])
        top5_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:5])
        
        top3_ratio = top3_heat / total_cluster_heat
        top5_ratio = top5_heat / total_cluster_heat
        
        print(f"[Heat Distribution] Top 3 clusters: {top3_ratio:.2%} (limit: {self.config.TOP3_HEAT_LIMIT_RATIO:.0%})")
        print(f"[Heat Distribution] Top 5 clusters: {top5_ratio:.2%} (limit: {self.config.TOP5_HEAT_LIMIT_RATIO:.0%})")
        
        # 检查是否超过阈值
        need_recycle = False
        if top3_ratio > self.config.TOP3_HEAT_LIMIT_RATIO:
            print(f"[Heat Distribution] Warning: Top 3 clusters exceed limit by {(top3_ratio - self.config.TOP3_HEAT_LIMIT_RATIO):.2%}")
            need_recycle = True
        
        if top5_ratio > self.config.TOP5_HEAT_LIMIT_RATIO:
            print(f"[Heat Distribution] Warning: Top 5 clusters exceed limit by {(top5_ratio - self.config.TOP5_HEAT_LIMIT_RATIO):.2%}")
            need_recycle = True
        
        if not need_recycle:
            print(f"[Heat Distribution] Heat distribution is within limits")
            return
        
        # 执行热力回收
        print(f"[Heat Distribution] Starting heat redistribution...")
        self._redistribute_cluster_heat(cluster_heat_list, total_cluster_heat)
        
        # 记录回收轮数
        self.last_heat_recycle_turn = self.current_turn
        self.heat_recycle_count += 1
        self.stats['heat_redistributions'] = self.stats.get('heat_redistributions', 0) + 1
        
        print(f"[Heat Distribution] Heat redistribution completed at turn {self.current_turn}")
    
    def _redistribute_cluster_heat(self, cluster_heat_list: List[Dict], total_cluster_heat: int):
        """重新分配簇热力，回收超过限制的部分"""
        # 计算每个簇的期望热力（基于其大小比例）
        total_size = sum(cluster['size'] for cluster in cluster_heat_list)
        if total_size == 0:
            return
        
        # 线性均匀扣除：计算需要回收的总热力
        excess_heat = 0
        
        # 计算前3簇的超额热力
        top3_excess = max(0, sum(cluster['heat'] for cluster in cluster_heat_list[:3]) - 
                         total_cluster_heat * self.config.TOP3_HEAT_LIMIT_RATIO)
        
        # 计算前5簇的超额热力
        top5_excess = max(0, sum(cluster['heat'] for cluster in cluster_heat_list[:5]) - 
                         total_cluster_heat * self.config.TOP5_HEAT_LIMIT_RATIO)
        
        excess_heat = max(top3_excess, top5_excess)
        
        if excess_heat <= 0:
            return
        
        print(f"[Heat Distribution] Excess heat to recycle: {excess_heat}")
        
        # 计算每个簇应回收的热力（按当前热力比例）
        total_top_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:5])
        if total_top_heat == 0:
            return
        
        recycled_heat = 0
        
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            # 从超额的簇中回收热力
            for i, cluster_info in enumerate(cluster_heat_list[:5]):
                cluster_id = cluster_info['cluster_id']
                cluster = self.clusters.get(cluster_id)
                if not cluster:
                    continue
                
                # 计算该簇应回收的热力（按比例）
                cluster_excess_ratio = cluster.total_heat / total_top_heat
                cluster_excess_heat = int(excess_heat * cluster_excess_ratio * self.config.HEAT_RECYCLE_RATE)
                
                # 确保不会回收过多，保留最小热力
                min_heat_for_cluster = max(self.config.MIN_CLUSTER_HEAT_AFTER_RECYCLE, 
                                         cluster.size * 10)  # 每个记忆至少10热力
                
                if cluster.total_heat - cluster_excess_heat < min_heat_for_cluster:
                    cluster_excess_heat = max(0, cluster.total_heat - min_heat_for_cluster)
                
                if cluster_excess_heat <= 0:
                    continue
                
                print(f"[Heat Distribution] Recycling {cluster_excess_heat} heat from cluster {cluster_id}")
                
                # 从簇内记忆回收热力（均匀扣除）
                memories_to_adjust = []
                for memory_id in list(cluster.memory_ids):
                    if memory_id in self.hot_memories:
                        memories_to_adjust.append(self.hot_memories[memory_id])
                
                if not memories_to_adjust:
                    continue
                
                # 计算每个记忆应扣除的热力
                heat_per_memory = max(1, cluster_excess_heat // len(memories_to_adjust))
                
                for memory in memories_to_adjust:
                    heat_to_deduct = min(heat_per_memory, memory.heat - 1)  # 至少保留1热力
                    if heat_to_deduct <= 0:
                        continue
                    
                    new_heat = memory.heat - heat_to_deduct
                    tx.add_memory_heat_update(memory.id, memory.heat, new_heat, cluster_id)
                    memory.heat = new_heat
                    memory.update_count += 1
                    
                    # 更新数据库
                    self.cursor.execute(f"""
                        UPDATE {self.config.MEMORY_TABLE}
                        SET heat = ?, update_count = update_count + 1
                        WHERE id = ?
                    """, (new_heat, memory.id))
                    
                    recycled_heat += heat_to_deduct
                
                # 更新簇总热力
                cluster.total_heat -= cluster_excess_heat
                tx.add_cluster_heat_update(cluster_id, -cluster_excess_heat)
                
                # 更新数据库中的簇热力
                self.cursor.execute(f"""
                    UPDATE {self.config.CLUSTER_TABLE}
                    SET total_heat = ?
                    WHERE id = ?
                """, (cluster.total_heat, cluster_id))
            
            # 将回收的热力放回热力池
            if recycled_heat > 0:
                with self.heat_pool_lock:
                    old_heat_pool = self.heat_pool
                    self.heat_pool += recycled_heat
                    
                    # 更新数据库中的热力池
                    self.cursor.execute(f"""
                        UPDATE {self.config.HEAT_POOL_TABLE}
                        SET heat_pool = ?
                        WHERE id = 1
                    """, (self.heat_pool,))
                
                print(f"[Heat Distribution] Recycled {recycled_heat} heat back to pool. "
                      f"Pool: {old_heat_pool} -> {self.heat_pool}")
                
                self.stats['heat_recycled_to_pool'] = self.stats.get('heat_recycled_to_pool', 0) + recycled_heat
        
        # 清除相关缓存
        for cluster_info in cluster_heat_list[:5]:
            cluster_id = cluster_info['cluster_id']
            self.cluster_search_cache.clear(cluster_id)
        
        # 使向量缓存失效（因为热力已改变）
        self.invalidate_vector_cache()
    
    def _is_in_suppression_period(self) -> bool:
        """检查是否处于热力回收抑制期"""
        if self.last_heat_recycle_turn == 0:
            return False
        
        turns_since_recycle = self.current_turn - self.last_heat_recycle_turn
        return turns_since_recycle < self.config.HEAT_RECYCLE_SUPPRESSION_TURNS
    
    def _get_suppression_factor(self) -> float:
        """获取当前抑制系数"""
        if not self._is_in_suppression_period():
            return 1.0
        
        # 线性衰减：距离回收时间越近，抑制越强
        turns_since_recycle = self.current_turn - self.last_heat_recycle_turn
        remaining_suppression = max(0, self.config.HEAT_RECYCLE_SUPPRESSION_TURNS - turns_since_recycle)
        
        # 线性插值：从HEAT_SUPPRESSION_FACTOR到1.0
        suppression_factor = self.config.HEAT_SUPPRESSION_FACTOR + (
            (1.0 - self.config.HEAT_SUPPRESSION_FACTOR) * 
            (1.0 - remaining_suppression / self.config.HEAT_RECYCLE_SUPPRESSION_TURNS)
        )
        
        return min(1.0, max(self.config.HEAT_SUPPRESSION_FACTOR, suppression_factor))
    
    def _cleanup_access_frequency_stats(self):
        """清理过期的访问频率统计"""
        with self.frequency_stats_lock:
            # 移除已经不存在的记忆的统计
            memory_ids_to_remove = []
            for memory_id in self.access_frequency_stats:
                if memory_id not in self.hot_memories:
                    memory_ids_to_remove.append(memory_id)
            
            for memory_id in memory_ids_to_remove:
                del self.access_frequency_stats[memory_id]
    
    def _cleanup_cluster_heat_history(self):
        """清理过期的簇热力历史记录"""
        max_history_age = 1000  # 最多保存1000轮的历史记录
        clusters_to_remove = []
        
        for cluster_id, history in self.cluster_heat_history.items():
            # 移除过期的记录
            fresh_history = [(turn, heat) for turn, heat in history 
                           if self.current_turn - turn <= max_history_age]
            
            if fresh_history:
                self.cluster_heat_history[cluster_id] = fresh_history
            else:
                clusters_to_remove.append(cluster_id)
        
        for cluster_id in clusters_to_remove:
            del self.cluster_heat_history[cluster_id]
    
    def _flush_update_queue(self):
        """处理所有待处理更新"""
        batch = []
        try:
            while True:
                item = self.update_queue.get_nowait()
                batch.append(item)
        except queue.Empty:
            pass
        
        if batch:
            self._process_batch_updates(batch)
    
    def _create_checkpoint_if_needed(self):
        """创建检查点 - 基于事件触发"""
        if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
            self._create_checkpoint()
    
    def _create_checkpoint(self):
        """创建检查点"""
        print(f"[Memory System] Creating system checkpoint (Turn: {self.current_turn})")
        
        try:
            # 保存所有热区记忆
            for memory in self.hot_memories.values():
                self.cursor.execute(f"""
                    UPDATE {self.config.MEMORY_TABLE}
                    SET heat = ?, last_interaction_turn = ?, access_count = ?, 
                        is_hot = ?, is_sleeping = ?, version = version + 1,
                        update_count = update_count + 1
                    WHERE id = ?
                """, (
                    memory.heat,
                    memory.last_interaction_turn,
                    memory.access_count,
                    int(memory.is_hot),
                    int(memory.is_sleeping),
                    memory.id
                ))
            
            # 保存簇状态
            for cluster in self.clusters.values():
                self.cursor.execute(f"""
                    UPDATE {self.config.CLUSTER_TABLE}
                    SET centroid = ?, total_heat = ?, hot_memory_count = ?, cold_memory_count = ?,
                        size = ?, last_updated_turn = ?, version = version + 1,
                        memory_additions_since_last_update = ?
                    WHERE id = ?
                """, (
                    self._vector_to_blob(cluster.centroid),
                    cluster.total_heat,
                    cluster.hot_memory_count,
                    cluster.cold_memory_count,
                    cluster.size,
                    self.current_turn,
                    cluster.memory_additions_since_last_update,
                    cluster.id
                ))
            
            # 保存热力池和当前轮数
            with self.heat_pool_lock:
                self.cursor.execute(f"""
                    UPDATE {self.config.HEAT_POOL_TABLE}
                    SET heat_pool = ?, total_allocated_heat = ?, version = version + 1,
                        current_turn = ?
                    WHERE id = 1
                """, (self.heat_pool, self.total_allocated_heat, self.current_turn))
            
            self.conn.commit()
            
            # 重置计数器
            self.memory_addition_count = 0
            
            print(f"[Memory System] Checkpoint created successfully")
        except Exception as e:
            print(f"[Memory System] Error creating checkpoint: {e}")
            self.conn.rollback()
    
    def _update_memory_cache_state(self):
        """更新内存缓存状态"""
        # 这里可以添加内存使用情况的检查和优化
        pass
    
    def _begin_transaction(self, transaction_id: str):
        """开始事务"""
        self.cursor.execute("BEGIN TRANSACTION")
        self.operation_log.append({
            'transaction_id': transaction_id,
            'type': 'begin',
            'turn': self.current_turn
        })
    
    def _commit_transaction(self, transaction_id: str, operations: List[Dict]) -> bool:
        """提交事务"""
        try:
            # 验证一致性约束
            if not self._validate_transaction(operations):
                return False
            
            # 应用所有操作
            for op in operations:
                self._apply_operation(op, immediate=True)
            
            self.conn.commit()
            
            self.operation_log.append({
                'transaction_id': transaction_id,
                'type': 'commit',
                'turn': self.current_turn,
                'operation_count': len(operations)
            })
            
            return True
        except Exception as e:
            print(f"Transaction commit failed: {e}")
            return False
    
    def _rollback_transaction(self, transaction_id: str):
        """回滚事务"""
        self.conn.rollback()
        self.operation_log.append({
            'transaction_id': transaction_id,
            'type': 'rollback',
            'turn': self.current_turn
        })
    
    def _validate_transaction(self, operations: List[Dict]) -> bool:
        """验证事务的一致性约束"""
        # 检查热力守恒
        total_heat_delta = 0
        cluster_heat_deltas = defaultdict(int)
        
        for op in operations:
            if op['type'] == OperationType.MEMORY_HEAT_UPDATE:
                heat_delta = op['new_heat'] - op['old_heat']
                total_heat_delta += heat_delta
                
                if op['cluster_id']:
                    cluster_heat_deltas[op['cluster_id']] += heat_delta
        
        # 热力必须守恒（除了从热力池分配或回收到热力池的情况）
        # 这里简化处理，实际应该更严格
        return True
    
    def _apply_operation(self, operation: Dict, immediate: bool = False):
        """应用单个操作"""
        op_type = operation['type']
        
        try:
            if op_type == OperationType.MEMORY_HEAT_UPDATE:
                self._apply_memory_heat_update(operation, immediate)
            elif op_type == OperationType.CLUSTER_HEAT_UPDATE:
                self._apply_cluster_heat_update(operation, immediate)
            
            # 记录操作到数据库
            if immediate:
                self._log_operation(operation, applied=True)
        except Exception as e:
            print(f"Failed to apply operation {operation}: {e}")
            raise
    
    def _apply_memory_heat_update(self, operation: Dict, immediate: bool):
        """应用记忆热力更新"""
        memory_id = operation['memory_id']
        new_heat = operation['new_heat']
        cluster_id = operation['cluster_id']
        
        # 获取内存锁
        with self.lock_manager.with_lock(f"memory_{memory_id}", self.config.MEMORY_LOCK_TIMEOUT):
            # 更新内存中的记忆热力
            if memory_id in self.hot_memories:
                memory = self.hot_memories[memory_id]
                memory.heat = new_heat
                memory.version += 1
                memory.update_count += 1
            
            # 更新数据库
            if immediate:
                self.cursor.execute(f"""
                    UPDATE {self.config.MEMORY_TABLE}
                    SET heat = ?, version = version + 1, update_count = update_count + 1
                    WHERE id = ?
                """, (new_heat, memory_id))
            
            # 更新簇热力（如果需要）
            if cluster_id and cluster_id in self.clusters:
                heat_delta = new_heat - operation['old_heat']
                if heat_delta != 0:
                    self._update_cluster_heat(cluster_id, heat_delta, immediate)
        
        # 使权重缓存失效
        if memory_id in self.weight_cache:
            del self.weight_cache[memory_id]
    
    def _apply_cluster_heat_update(self, operation: Dict, immediate: bool):
        """应用簇热力更新"""
        cluster_id = operation['cluster_id']
        heat_delta = operation['heat_delta']
        
        # 获取簇锁
        with self.lock_manager.with_lock(f"cluster_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT):
            if cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                
                if immediate:
                    # 立即更新
                    cluster.total_heat += heat_delta
                    cluster.version += 1
                    
                    # 更新数据库
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET total_heat = total_heat + ?, version = version + 1
                        WHERE id = ?
                    """, (heat_delta, cluster_id))
                else:
                    # 延迟更新，先记录到pending
                    cluster.pending_heat_delta += heat_delta
        
        # 使权重缓存失效（因为簇总热力已改变）
        self.weight_cache.clear()
    
    def _log_operation(self, operation: Dict, applied: bool = False):
        """记录操作到日志"""
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
    
    def _update_cluster_heat(self, cluster_id: str, heat_delta: int, immediate: bool = True):
        """更新簇热力（封装方法）"""
        if immediate:
            # 同步更新
            with self.lock_manager.with_lock(f"cluster_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT):
                if cluster_id in self.clusters:
                    self.clusters[cluster_id].total_heat += heat_delta
                
                # 更新数据库
                self.cursor.execute(f"""
                    UPDATE {self.config.CLUSTER_TABLE}
                    SET total_heat = total_heat + ?
                    WHERE id = ?
                """, (heat_delta, cluster_id))
        else:
            # 异步更新，放入队列
            self.update_queue.put({
                'type': 'cluster_heat_update',
                'cluster_id': cluster_id,
                'heat_delta': heat_delta,
                'turn': self.current_turn
            })
        
        # 使权重缓存失效
        self.weight_cache.clear()
    
    def _update_memory_and_cluster_heat_atomic(self, memory_id: str, new_heat: int, 
                                              cluster_id: str = None, 
                                              old_heat: int = None) -> bool:
        """原子更新记忆和簇的热力"""
        if old_heat is None and memory_id in self.hot_memories:
            old_heat = self.hot_memories[memory_id].heat
        
        if old_heat is None:
            # 从数据库获取
            self.cursor.execute(f"SELECT heat FROM {self.config.MEMORY_TABLE} WHERE id = ?", (memory_id,))
            row = self.cursor.fetchone()
            old_heat = row['heat'] if row else 0
        
        heat_delta = new_heat - old_heat
        
        # 使用事务确保原子性
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            tx.add_memory_heat_update(memory_id, old_heat, new_heat, cluster_id)
            
            if cluster_id and heat_delta != 0:
                tx.add_cluster_heat_update(cluster_id, heat_delta)
        
        return True
    
    def _process_batch_updates(self, batch: List[Dict]):
        """批量处理更新"""
        # 按簇分组
        cluster_updates = defaultdict(int)
        cluster_centroid_updates = {}
        memory_updates = {}
        
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
        
        # 批量更新簇
        with self.conn:
            for cluster_id, heat_delta in cluster_updates.items():
                if heat_delta != 0:
                    # 更新数据库
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET total_heat = total_heat + ?, pending_heat_delta = pending_heat_delta - ?
                        WHERE id = ?
                    """, (heat_delta, heat_delta, cluster_id))
                    
                    # 更新内存
                    if cluster_id in self.clusters:
                        with self.clusters[cluster_id].lock:
                            self.clusters[cluster_id].total_heat += heat_delta
                            self.clusters[cluster_id].pending_heat_delta -= heat_delta
            
            # 批量更新簇质心
            if cluster_centroid_updates:
                for cluster_id, update_data in cluster_centroid_updates.items():
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET centroid = ?, last_updated_turn = ?, version = version + 1,
                            memory_additions_since_last_update = 0
                        WHERE id = ?
                    """, (
                        self._vector_to_blob(update_data['centroid']),
                        update_data['turn'],
                        cluster_id
                    ))
        
        # 批量更新记忆
        for memory_id, new_heat in memory_updates.items():
            if memory_id in self.hot_memories:
                self.hot_memories[memory_id].heat = new_heat
        
        # 使缓存失效
        if cluster_updates or memory_updates:
            self.weight_cache.clear()
    
    def _apply_pending_cluster_updates(self):
        """应用pending的簇更新"""
        pending_clusters = []
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.pending_heat_delta != 0:
                pending_clusters.append((cluster_id, cluster.pending_heat_delta))
        
        if pending_clusters:
            with self.conn:
                for cluster_id, pending_delta in pending_clusters:
                    # 更新数据库
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET total_heat = total_heat + ?, pending_heat_delta = 0
                        WHERE id = ?
                    """, (pending_delta, cluster_id))
                    
                    # 更新内存并清零pending
                    if cluster_id in self.clusters:
                        with self.clusters[cluster_id].lock:
                            self.clusters[cluster_id].total_heat += pending_delta
                            self.clusters[cluster_id].pending_heat_delta = 0
            
            # 使权重缓存失效
            self.weight_cache.clear()
    
    # =============== 重复检测方法 ===============
    def _check_duplicate(self, vector: np.ndarray, content: str = None) -> Optional[str]:
        """检查是否为重复记忆，返回已存在的记忆ID（如果找到的话）"""
        if not self.config.DUPLICATE_CHECK_ENABLED:
            return None
        
        best_similarity = 0.0
        best_memory_id = None
        best_memory = None
        
        # 1. 首先检查热区记忆
        for memory_id, memory in self.hot_memories.items():
            similarity = self._compute_similarity(vector, memory.vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_memory_id = memory_id
                best_memory = memory
            
            # 如果已经超过阈值，提前返回
            if best_similarity >= self.config.DUPLICATE_THRESHOLD:
                break
        
        # 2. 如果热区没找到高度相似的，检查内容文本（可选）
        if content and best_similarity < self.config.DUPLICATE_THRESHOLD:
            # 可以添加简单的文本比较（如哈希、关键词等）
            # 这里使用简单的文本哈希检查
            content_hash = hashlib.md5(content.strip().lower().encode()).hexdigest()
            for memory_id, memory in self.hot_memories.items():
                memory_hash = hashlib.md5(memory.content.strip().lower().encode()).hexdigest()
                if content_hash == memory_hash:
                    return memory_id
        
        # 3. 检查冷区（可选，需要时再实现）
        
        if best_similarity >= self.config.DUPLICATE_THRESHOLD:
            print(f"[Duplicate Detection] Found duplicate memory: {best_memory_id} with similarity {best_similarity:.4f}")
            return best_memory_id
        
        return None
    
    def enable_duplicate_detection(self, enabled: bool = True, threshold: float = None):
        """启用或禁用重复检测"""
        self.config.DUPLICATE_CHECK_ENABLED = enabled
        if threshold is not None:
            self.config.DUPLICATE_THRESHOLD = threshold
        
        print(f"[Duplicate Detection] {'Enabled' if enabled else 'Disabled'} with threshold {self.config.DUPLICATE_THRESHOLD}")
    
    # =============== 簇质心更新方法 ===============
    
    def _update_cluster_centroids_batch(self):
        """批量更新簇质心（基于记忆存入次数）"""
        if self.memory_additions_since_last_centroid_update < self.config.CENTROID_UPDATE_FREQUENCY:
            return
        
        print(f"[Memory System] Updating cluster centroids after {self.memory_additions_since_last_centroid_update} memory additions")
        
        # 重置计数器
        self.memory_additions_since_last_centroid_update = 0
        
        # 收集需要更新质心的簇
        clusters_to_update = list(self.clusters_needing_centroid_update)
        self.clusters_needing_centroid_update.clear()
        
        if not clusters_to_update:
            # 选择最近有新增记忆的簇
            for cluster_id, cluster in self.clusters.items():
                if cluster.memory_additions_since_last_update > 0:
                    clusters_to_update.append(cluster_id)
        
        # 批量处理
        batch_size = self.config.CENTROID_UPDATE_BATCH_SIZE
        for i in range(0, len(clusters_to_update), batch_size):
            batch = clusters_to_update[i:i+batch_size]
            self._update_cluster_centroids(batch)
    
    def _update_cluster_centroids(self, cluster_ids: List[str]):
        """更新指定簇的质心"""
        centroid_updates = {}
        
        for cluster_id in cluster_ids:
            if cluster_id not in self.clusters:
                continue
            
            cluster = self.clusters[cluster_id]
            if cluster.memory_additions_since_last_update == 0 and not cluster.pending_centroid_updates:
                continue
            
            # 获取簇锁
            with cluster.lock:
                if cluster.memory_additions_since_last_update >= self.config.CENTROID_FULL_RECALC_THRESHOLD:
                    # 簇变化太大，完全重算质心
                    new_centroid = self._recalculate_cluster_centroid(cluster_id)
                    self.stats['full_centroid_recalculations'] += 1
                else:
                    # 增量更新质心
                    new_centroid = self._incremental_update_cluster_centroid(cluster)
                
                if new_centroid is not None:
                    # 更新内存中的质心
                    cluster.centroid = new_centroid
                    cluster.memory_additions_since_last_update = 0
                    cluster.pending_centroid_updates.clear()
                    cluster.last_updated_turn = self.current_turn
                    cluster.version += 1
                    
                    # 更新向量缓存
                    self.cluster_vectors[cluster_id] = new_centroid
                    
                    # 记录更新
                    centroid_updates[cluster_id] = {
                        'centroid': new_centroid,
                        'turn': self.current_turn
                    }
        
        # 批量更新数据库
        if centroid_updates:
            with self.conn:
                for cluster_id, update_data in centroid_updates.items():
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET centroid = ?, last_updated_turn = ?, version = version + 1,
                            memory_additions_since_last_update = 0
                        WHERE id = ?
                    """, (
                        self._vector_to_blob(update_data['centroid']),
                        update_data['turn'],
                        cluster_id
                    ))
            
            self.stats['centroid_updates'] += len(centroid_updates)
            print(f"[Memory System] Updated centroids for {len(centroid_updates)} clusters")
            
            # 清除簇搜索缓存
            for cluster_id in cluster_ids:
                self.cluster_search_cache.clear(cluster_id)
    
    def _incremental_update_cluster_centroid(self, cluster: SemanticCluster) -> Optional[np.ndarray]:
        """增量更新簇质心"""
        if not cluster.pending_centroid_updates and cluster.memory_additions_since_last_update == 0:
            return None
        
        # 如果簇大小很小或者没有pending更新，使用增量公式
        if cluster.pending_centroid_updates:
            # 使用pending更新计算新质心
            new_centroid = cluster.centroid.copy()
            
            for vector, add in cluster.pending_centroid_updates:
                if add:
                    # 添加记忆：新质心 = (旧质心 * 旧大小 + 新向量) / (旧大小 + 1)
                    if cluster.size > 0:
                        new_centroid = (new_centroid * cluster.size + vector) / (cluster.size + 1)
                    else:
                        new_centroid = vector
                    cluster.size += 1
                else:
                    # 移除记忆：新质心 = (旧质心 * 旧大小 - 移除向量) / (旧大小 - 1)
                    if cluster.size > 1:
                        new_centroid = (new_centroid * cluster.size - vector) / (cluster.size - 1)
                    else:
                        new_centroid = np.zeros(self.embedding_dim, dtype=np.float32)
                    cluster.size -= 1
            
            # 归一化
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            return new_centroid
        
        return None
    
    def _recalculate_cluster_centroid(self, cluster_id: str) -> Optional[np.ndarray]:
        """完全重算簇质心"""
        if cluster_id not in self.clusters:
            return None
        
        # 获取簇内所有热区记忆的向量
        self.cursor.execute(f"""
            SELECT vector FROM {self.config.MEMORY_TABLE}
            WHERE cluster_id = ? AND is_hot = 1
            LIMIT 1000
        """, (cluster_id,))
        
        rows = self.cursor.fetchall()
        if not rows:
            return None
        
        vectors = [self._blob_to_vector(row['vector']) for row in rows]
        
        # 计算平均值作为新质心
        new_centroid = np.mean(vectors, axis=0)
        
        # 归一化
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        
        return new_centroid
    
    def _schedule_cluster_centroid_update(self, cluster_id: str, vector: np.ndarray, add: bool = True):
        """调度簇质心更新"""
        if cluster_id not in self.clusters:
            return
        
        cluster = self.clusters[cluster_id]
        with cluster.lock:
            # 记录pending更新
            cluster.pending_centroid_updates.append((vector.copy(), add))
            
            # 更新计数器
            if add:
                cluster.memory_additions_since_last_update += 1
            
            # 标记需要更新
            self.clusters_needing_centroid_update.add(cluster_id)
        
        # 更新全局计数器
        if add:
            self.memory_additions_since_last_centroid_update += 1
        
        # 使向量缓存失效
        self.invalidate_vector_cache()
    
    # =============== 嵌入和相似度计算（支持外部函数） ===============
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入"""
        if self._external_embedding_func is not None:
            return self._external_embedding_func(text)
        elif self.model:
            try:
                # 禁用进度条
                return self.model.encode(text, show_progress_bar=False)
            except:
                # 如果模型不支持 show_progress_bar 参数
                return self.model.encode(text)
        else:
            # 使用随机向量作为回退
            if not hasattr(self, 'embedding_dim'):
                self.embedding_dim = self.config.EMBEDDING_DIM
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if self._external_similarity_func is not None:
            return self._external_similarity_func(vec1, vec2)
        
        # 使用内部实现
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    # =============== 访问频率和轮数相关方法 ===============
    
    def _update_access_frequency(self, memory_id: str):
        """更新访问频率统计"""
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
                
                # 只保留最近100次交互记录（避免内存占用过多）
                if len(stats['recent_interactions']) > 100:
                    stats['recent_interactions'] = stats['recent_interactions'][-100:]
                
                # 检查是否需要重置统计（例如每1000轮重置一次）
                if self.current_turn - stats['last_reset_turn'] > 1000:
                    stats['count'] = 1
                    stats['last_reset_turn'] = self.current_turn
                    stats['recent_interactions'] = [self.current_turn]
        
        # 使权重缓存失效
        if memory_id in self.weight_cache:
            del self.weight_cache[memory_id]
    
    def _get_access_frequency_weight(self, memory_id: str, memory_item: MemoryItem) -> float:
        """获取访问频率权重（过度访问会降低权重）"""
        with self.frequency_stats_lock:
            if memory_id not in self.access_frequency_stats:
                return 1.0
            
            stats = self.access_frequency_stats[memory_id]
            access_count = stats['count']
            
            # 计算基于轮数的访问频率（最近1000轮内的访问次数）
            recent_interactions = [turn for turn in stats['recent_interactions'] 
                                  if self.current_turn - turn < 1000]
            recent_count = len(recent_interactions)
            
            # 使用最近访问次数和总访问次数综合判断
            total_factor = min(1.0, self.config.ACCESS_FREQUENCY_DISCOUNT_THRESHOLD / max(1, access_count))
            recent_factor = min(1.0, self.config.ACCESS_FREQUENCY_DISCOUNT_THRESHOLD / max(1, recent_count))
            
            # 综合权重（更关注最近访问频率）
            weight = 0.3 * total_factor + 0.7 * recent_factor
            
            # 如果记忆被标记为休眠，进一步降低权重
            if memory_item.is_sleeping:
                weight *= 0.5
            
            return max(0.1, weight)  # 确保权重不低于0.1
    
    def _get_recency_weight(self, memory_item: MemoryItem) -> float:
        """获取最近访问权重（基于轮数间隔）"""
        # 计算距离上次交互的轮数间隔
        turns_since_interaction = self.current_turn - memory_item.last_interaction_turn
        
        # 使用线性衰减计算权重（避免指数衰减的过度惩罚）
        # 每轮衰减 config.RECENCY_WEIGHT_DECAY_PER_TURN
        weight = 1.0 - (turns_since_interaction * self.config.RECENCY_WEIGHT_DECAY_PER_TURN)
        
        return max(0.1, min(1.0, weight))
    
    def _get_relative_heat_weight(self, memory_item: MemoryItem, cluster_total_heat: int) -> float:
        """获取相对热力权重"""
        if cluster_total_heat <= 0:
            return 1.0
        
        # 计算相对热力比例
        relative_heat = memory_item.heat / cluster_total_heat
        
        # 使用幂函数平滑权重分布（避免热力过高的记忆主导搜索结果）
        # 当 RELATIVE_HEAT_WEIGHT_POWER < 1 时，会压缩高权重记忆的优势
        weight = relative_heat ** self.config.RELATIVE_HEAT_WEIGHT_POWER
        
        # 添加最小权重保证
        weight = max(0.1, min(1.0, weight))
        
        return weight
    
    # =============== 向量化优化方法 ===============
    
    def _rebuild_vector_cache(self):
        """重建向量缓存"""
        with self.vector_cache_lock:
            memory_ids = []
            vectors = []
            
            for memory_id, memory in self.hot_memories.items():
                memory_ids.append(memory_id)
                vectors.append(memory.vector)
            
            if vectors:
                self.vector_cache.vectors = np.array(vectors, dtype=np.float32)  # (M, d)
            else:
                self.vector_cache.vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
            
            self.vector_cache.memory_ids = memory_ids
            self.vector_cache.last_updated = time.time()
            self.vector_cache.is_valid = True
            
            # 清除预计算的归一化向量
            self._normalized_vectors = None
            self._precomputed_memory_norms = None
            
            print(f"[Vector Cache] Rebuilt cache with {len(memory_ids)} vectors")
    
    def _ensure_vector_cache(self):
        """确保向量缓存是最新的"""
        with self.vector_cache_lock:
            if (self.vector_cache.is_valid and 
                self.vector_cache.vectors is not None and
                len(self.vector_cache.memory_ids) == len(self.hot_memories)):
                return  # 缓存有效
            
            self._rebuild_vector_cache()
    
    def _compute_all_similarities_vectorized(self, query_vector: np.ndarray) -> np.ndarray:
        """向量化计算所有相似度"""
        # 1. 确保向量缓存有效
        self._ensure_vector_cache()
        
        vectors = self.vector_cache.vectors
        if vectors.shape[0] == 0:
            return np.array([])
        
        # 2. 归一化查询向量（一次性）
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(vectors.shape[0])
        
        normalized_query = query_vector / query_norm
        
        # 3. 预归一化所有记忆向量（缓存）
        if self._normalized_vectors is None or self._precomputed_memory_norms is None:
            # 计算并缓存记忆向量的范数
            memory_norms = np.linalg.norm(vectors, axis=1)
            # 避免除零
            memory_norms[memory_norms == 0] = 1e-10
            self._normalized_vectors = vectors / memory_norms[:, np.newaxis]
            self._precomputed_memory_norms = memory_norms
        
        # 4. 向量化点积计算（单次矩阵运算）
        similarities = np.dot(self._normalized_vectors, normalized_query)
        
        # 5. 确保相似度在有效范围内
        similarities = np.clip(similarities, -1.0, 1.0)
        
        return similarities
    
    def _get_cached_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """获取缓存的相似度（计算或从缓存获取）"""
        # 1. 检查缓存
        cached_similarities = self.similarity_cache.get(query_vector)
        if cached_similarities is not None:
            self.stats['similarity_cache_hits'] += 1
            return cached_similarities
        
        self.stats['similarity_cache_misses'] += 1
        
        # 2. 计算并缓存
        similarities = self._compute_all_similarities_vectorized(query_vector)
        self.similarity_cache.put(query_vector, similarities)
        
        return similarities
    
    def _ensure_weight_cache(self):
        """确保权重缓存是最新的（基于轮数）"""
        current_turn = self.current_turn
        
        # 每100轮或记忆数量变化时更新权重缓存
        if (current_turn - self.weight_cache_turn > 100 or
            len(self.weight_cache) != len(self.hot_memories)):
            
            with self.frequency_stats_lock:
                self.weight_cache.clear()
                
                for memory_id, memory in self.hot_memories.items():
                    cluster_total_heat = self.clusters[memory.cluster_id].total_heat if memory.cluster_id in self.clusters else 1
                    
                    # 预计算常用权重
                    self.weight_cache[memory_id] = {
                        'relative_heat_weight': self._get_relative_heat_weight(memory, cluster_total_heat),
                        'access_frequency_weight': self._get_access_frequency_weight(memory_id, memory),
                        'recency_weight': self._get_recency_weight(memory),
                        'heat': memory.heat,
                        'last_updated_turn': current_turn
                    }
            
            self.weight_cache_turn = current_turn
            print(f"[Weight Cache] Updated cache for {len(self.weight_cache)} memories (turn: {current_turn})")
    
    def invalidate_vector_cache(self, memory_id: str = None, operation: str = 'update'):
        """使向量缓存失效"""
        with self.vector_cache_lock:
            if memory_id and self.vector_cache.is_valid and self.vector_cache.vectors is not None:
                # 尝试增量更新
                if memory_id in self.vector_cache.memory_ids:
                    idx = self.vector_cache.memory_ids.index(memory_id)
                    memory = self.hot_memories.get(memory_id)
                    if memory:
                        self.vector_cache.vectors[idx] = memory.vector
                        
                        # 使预计算的归一化向量失效
                        self._normalized_vectors = None
                        self._precomputed_memory_norms = None
            else:
                # 完全失效
                self.vector_cache.is_valid = False
        
        # 清除相似度缓存
        self.similarity_cache.cache.clear()
    
    # =============== 分层搜索方法（向量化优化版） ===============
    
    def search_layered_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                               max_total_results: int = None,
                               config_override: Dict = None) -> Dict[str, LayeredSearchResult]:
        """优化的分层搜索 - 使用向量化和缓存"""
        if not self.config.LAYERED_SEARCH_ENABLED:
            warnings.warn("Layered search is disabled. Using default search.")
            return self._fallback_search(query_text, query_vector, max_total_results)
        
        if query_vector is None and query_text is not None:
            query_vector = self._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")
        
        # 使用配置覆盖或默认配置
        config = config_override or self.config.LAYERED_SEARCH_CONFIG
        
        if max_total_results is None:
            max_total_results = self.config.LAYERED_SEARCH_MAX_TOTAL_RESULTS
        
        # 记录搜索统计
        self.stats['layered_searches'] += 1
        self.stats['vectorized_searches'] += 1
        
        # ========= 优化1：向量化相似度计算 =========
        start_time = time.time()
        similarities = self._get_cached_similarities(query_vector)
        sim_compute_time = time.time() - start_time
        
        # ========= 优化2：预计算权重缓存 =========
        self._ensure_weight_cache()
        
        # ========= 优化3：向量化分层筛选 =========
        memory_ids = self.vector_cache.memory_ids
        layered_results = {}
        
        # 为去重准备集合
        seen_memory_ids = set() if self.config.LAYERED_SEARCH_DEDUPLICATE else None
        
        # 按层处理（从高相似度到低相似度）
        for layer_name in ["layer_3", "layer_2", "layer_1"]:
            if layer_name not in config:
                continue
            
            layer_config = config[layer_name]
            similarity_min, similarity_max = layer_config["similarity_range"]
            max_results = layer_config["max_results"]
            min_heat = layer_config.get("min_heat_required", 0)
            
            # 向量化筛选：找到在范围内的记忆索引
            mask = (similarities >= similarity_min) & (similarities < similarity_max)
            candidate_indices = np.where(mask)[0]
            
            if candidate_indices.size == 0:
                # 该层没有结果
                layered_results[layer_name] = LayeredSearchResult(
                    layer_name=layer_name,
                    similarity_range=(similarity_min, similarity_max),
                    results=[],
                    achieved_count=0,
                    target_count=max_results,
                    avg_similarity=0.0,
                    avg_final_score=0.0
                )
                continue
            
            # 如果启用去重，过滤已选中的记忆
            if seen_memory_ids is not None and candidate_indices.size > 0:
                valid_indices = []
                for idx in candidate_indices:
                    memory_id = memory_ids[idx]
                    if memory_id not in seen_memory_ids:
                        valid_indices.append(idx)
                candidate_indices = np.array(valid_indices)
            
            # ========= 优化4：批量处理候选记忆 =========
            layer_results = []
            candidates_processed = 0
            
            # 按相似度排序（只取前N个）
            sorted_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]
            
            for idx in sorted_indices[:max_results]:
                memory_id = memory_ids[idx]
                memory = self.hot_memories[memory_id]
                
                # 检查最低热力要求
                if memory.heat < min_heat:
                    continue
                
                # 从权重缓存获取权重（避免重复计算）
                cached_weights = self.weight_cache.get(memory_id, {})
                if not cached_weights:
                    # 缓存未命中，重新计算
                    cluster_total_heat = self.clusters[memory.cluster_id].total_heat if memory.cluster_id in self.clusters else 1
                    
                    relative_heat_weight = self._get_relative_heat_weight(memory, cluster_total_heat)
                    access_frequency_weight = self._get_access_frequency_weight(memory_id, memory)
                    recency_weight = self._get_recency_weight(memory)
                else:
                    relative_heat_weight = cached_weights['relative_heat_weight']
                    access_frequency_weight = cached_weights['access_frequency_weight']
                    recency_weight = cached_weights['recency_weight']
                
                # 应用层特定的权重系数
                heat_weight_factor = layer_config.get("heat_weight_factor", 1.0)
                frequency_weight_factor = layer_config.get("frequency_weight_factor", 1.0)
                recency_weight_factor = layer_config.get("recency_weight_factor", 1.0)
                base_score_factor = layer_config.get("base_score_factor", 1.0)
                
                # 调整后的权重
                adj_relative_heat_weight = relative_heat_weight * heat_weight_factor
                adj_access_frequency_weight = access_frequency_weight * frequency_weight_factor
                adj_recency_weight = recency_weight * recency_weight_factor
                adj_base_similarity = similarities[idx] * base_score_factor
                
                # 最终得分（加权几何平均）
                weights = [adj_relative_heat_weight, adj_access_frequency_weight, adj_recency_weight]
                weights_nonzero = [max(0.0001, w) for w in weights]
                geometric_mean = np.exp(np.mean(np.log(weights_nonzero)))
                
                final_score = adj_base_similarity * geometric_mean
                
                # 创建结果对象
                result = WeightedMemoryResult(
                    memory=memory,
                    base_similarity=similarities[idx],
                    relative_heat_weight=adj_relative_heat_weight,
                    access_frequency_weight=adj_access_frequency_weight,
                    recency_weight=adj_recency_weight,
                    final_score=final_score,
                    ranking_position=len(layer_results) + 1
                )
                
                layer_results.append(result)
                candidates_processed += 1
                
                # 记录已选记忆ID
                if seen_memory_ids is not None:
                    seen_memory_ids.add(memory_id)
                
                # 如果达到该层目标数量，提前退出
                if candidates_processed >= max_results:
                    break
            
            # 计算统计信息
            achieved_count = len(layer_results)
            if achieved_count > 0:
                avg_similarity = np.mean([r.base_similarity for r in layer_results])
                avg_final_score = np.mean([r.final_score for r in layer_results])
            else:
                avg_similarity = 0.0
                avg_final_score = 0.0
            
            # 存储层结果
            layered_results[layer_name] = LayeredSearchResult(
                layer_name=layer_name,
                similarity_range=(similarity_min, similarity_max),
                results=layer_results,
                achieved_count=achieved_count,
                target_count=max_results,
                avg_similarity=avg_similarity,
                avg_final_score=avg_final_score
            )
            
            # 如果已收集足够总结果，提前退出
            total_results = sum(len(r.results) for r in layered_results.values())
            if total_results >= max_total_results:
                break
        
        # 添加性能统计
        total_time = time.time() - start_time
        if total_time > 0.1:  # 超过100ms时记录警告
            print(f"[Performance] Vectorized layered search: {total_time*1000:.1f}ms "
                  f"(similarity: {sim_compute_time*1000:.1f}ms)")
        
        return layered_results
    
    def _apply_fallback(self, layered_results: Dict[str, LayeredSearchResult],
                       config: Dict, query_vector: np.ndarray,
                       max_total_results: int, exclude_ids: Set[str]):
        """应用后备策略：当某些层结果不足时，用其他层补充"""
        total_results = sum(len(r.results) for r in layered_results.values())
        
        if total_results >= max_total_results:
            return
        
        # 确定需要补充的结果数
        needed_results = max_total_results - total_results
        
        # 找出结果不足的层
        deficient_layers = []
        for layer_name, result in layered_results.items():
            if result.achieved_count < result.target_count:
                deficit = result.target_count - result.achieved_count
                deficient_layers.append((layer_name, deficit, result.similarity_range))
        
        # 如果没有不足的层，直接返回
        if not deficient_layers:
            return
        
        # 按不足程度排序（最不足的优先）
        deficient_layers.sort(key=lambda x: x[1], reverse=True)
        
        # 后备搜索：从相似度稍低的层中寻找补充
        for layer_name, deficit, (similarity_min, similarity_max) in deficient_layers:
            if needed_results <= 0:
                break
            
            # 计算后备搜索范围（降低相似度要求）
            fallback_min = max(0.5, similarity_min - 0.1)  # 至少0.5相似度
            fallback_max = similarity_max
            
            # 在更宽泛的范围内搜索
            fallback_results = self._search_within_layer_legacy(
                query_vector=query_vector,
                similarity_min=fallback_min,
                similarity_max=fallback_max,
                max_results=deficit,
                min_heat=0,  # 后备搜索降低热力要求
                layer_config=config.get(layer_name, {}),
                exclude_ids=exclude_ids
            )
            
            # 添加后备结果
            if fallback_results:
                current_results = layered_results[layer_name].results
                current_results.extend(fallback_results)
                
                # 重新排序
                current_results.sort(key=lambda x: x.final_score, reverse=True)
                current_results = current_results[:layered_results[layer_name].target_count]
                
                # 更新统计
                achieved_count = len(current_results)
                if achieved_count > 0:
                    avg_similarity = sum(r.base_similarity for r in current_results) / achieved_count
                    avg_final_score = sum(r.final_score for r in current_results) / achieved_count
                else:
                    avg_similarity = 0.0
                    avg_final_score = 0.0
                
                # 更新层结果
                layered_results[layer_name] = LayeredSearchResult(
                    layer_name=layer_name,
                    similarity_range=layered_results[layer_name].similarity_range,
                    results=current_results,
                    achieved_count=achieved_count,
                    target_count=layered_results[layer_name].target_count,
                    avg_similarity=avg_similarity,
                    avg_final_score=avg_final_score
                )
                
                # 更新已见ID集合
                if exclude_ids is not None:
                    for result in fallback_results:
                        exclude_ids.add(result.memory.id)
                
                needed_results -= len(fallback_results)
    
    def _search_within_layer_legacy(self, query_vector: np.ndarray,
                                   similarity_min: float, similarity_max: float,
                                   max_results: int, min_heat: int,
                                   layer_config: Dict,
                                   exclude_ids: Set[str] = None) -> List[WeightedMemoryResult]:
        """传统方法在特定相似度层内搜索记忆（用于后备搜索）"""
        candidates = []
        
        # 遍历热区记忆
        for memory_id, memory in self.hot_memories.items():
            # 排除条件
            if memory.is_sleeping:
                continue
            
            if exclude_ids and memory_id in exclude_ids:
                continue
            
            if memory.heat < min_heat:
                continue
            
            # 计算相似度
            similarity = self._compute_similarity(query_vector, memory.vector)
            
            # 检查是否在目标层内
            if similarity_min <= similarity < similarity_max:
                # 计算各种权重
                cluster_total_heat = self.clusters[memory.cluster_id].total_heat if memory.cluster_id in self.clusters else 1
                
                relative_heat_weight = self._get_relative_heat_weight(memory, cluster_total_heat)
                access_frequency_weight = self._get_access_frequency_weight(memory_id, memory)
                recency_weight = self._get_recency_weight(memory)
                
                # 应用层特定的权重系数
                heat_weight_factor = layer_config.get("heat_weight_factor", 1.0)
                frequency_weight_factor = layer_config.get("frequency_weight_factor", 1.0)
                recency_weight_factor = layer_config.get("recency_weight_factor", 1.0)
                base_score_factor = layer_config.get("base_score_factor", 1.0)
                
                # 调整后的权重
                adj_relative_heat_weight = (relative_heat_weight * heat_weight_factor)
                adj_access_frequency_weight = (access_frequency_weight * frequency_weight_factor)
                adj_recency_weight = (recency_weight * recency_weight_factor)
                adj_base_similarity = similarity * base_score_factor
                
                # 最终得分（使用加权几何平均）
                weights = [adj_relative_heat_weight, adj_access_frequency_weight, adj_recency_weight]
                geometric_mean = np.exp(np.mean(np.log([max(0.0001, w) for w in weights])))
                
                final_score = adj_base_similarity * geometric_mean
                
                # 创建结果对象
                result = WeightedMemoryResult(
                    memory=memory,
                    base_similarity=similarity,
                    relative_heat_weight=adj_relative_heat_weight,
                    access_frequency_weight=adj_access_frequency_weight,
                    recency_weight=adj_recency_weight,
                    final_score=final_score,
                    ranking_position=0
                )
                
                candidates.append((final_score, result))
        
        # 按最终得分排序并限制数量
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_candidates = candidates[:max_results]
        
        # 创建最终结果列表
        results = [result for _, result in selected_candidates]
        
        # 设置排名
        for i, result in enumerate(results):
            result.ranking_position = i + 1
        
        return results
    
    def _fallback_search(self, query_text: str = None, query_vector: np.ndarray = None,
                        max_total_results: int = None) -> Dict[str, LayeredSearchResult]:
        """后备搜索方法（当分层搜索被禁用时）"""
        if query_vector is None and query_text is not None:
            query_vector = self._get_embedding(query_text)
        
        if max_total_results is None:
            max_total_results = 8
        
        # 使用默认搜索
        all_results = self.search_similar_memories(
            query_vector=query_vector,
            max_results=max_total_results,
            use_weighting=True
        )
        
        # 将结果分配到虚拟层中（用于统一接口）
        layered_results = {}
        default_layers = {
            "layer_1": {"similarity_range": (0.0, 1.0), "results": [], "count": 0},
            "layer_2": {"similarity_range": (0.0, 1.0), "results": [], "count": 0},
            "layer_3": {"similarity_range": (0.0, 1.0), "results": [], "count": 0}
        }
        
        # 简单分配：按相似度分配到不同层
        for result in all_results:
            if result.base_similarity >= 0.85:
                default_layers["layer_3"]["results"].append(result)
            elif result.base_similarity >= 0.80:
                default_layers["layer_2"]["results"].append(result)
            elif result.base_similarity >= 0.75:
                default_layers["layer_1"]["results"].append(result)
        
        # 转换为LayeredSearchResult格式
        for layer_name, layer_data in default_layers.items():
            results = layer_data["results"]
            achieved_count = len(results)
            
            if achieved_count > 0:
                avg_similarity = sum(r.base_similarity for r in results) / achieved_count
                avg_final_score = sum(r.final_score for r in results) / achieved_count
            else:
                avg_similarity = 0.0
                avg_final_score = 0.0
            
            layered_results[layer_name] = LayeredSearchResult(
                layer_name=layer_name,
                similarity_range=layer_data["similarity_range"],
                results=results,
                achieved_count=achieved_count,
                target_count=0,  # 虚拟层没有目标数量
                avg_similarity=avg_similarity,
                avg_final_score=avg_final_score
            )
        
        return layered_results
    
    def get_layered_search_results(self, query_text: str = None, query_vector: np.ndarray = None,
                                  flatten_results: bool = True) -> List[WeightedMemoryResult]:
        """
        获取分层搜索结果（扁平化版本）
        
        参数:
            query_text: 查询文本
            query_vector: 查询向量
            flatten_results: 是否将分层结果扁平化为一个列表
            
        返回:
            如果flatten_results=True，返回扁平化的WeightedMemoryResult列表
            否则返回分层的LayeredSearchResult字典
        """
        layered_results = self.search_layered_memories(
            query_text=query_text,
            query_vector=query_vector
        )
        
        if not flatten_results:
            return layered_results
        
        # 扁平化结果：按照层优先级（layer_3 > layer_2 > layer_1）合并
        flattened = []
        for layer_name in ["layer_3", "layer_2", "layer_1"]:
            if layer_name in layered_results:
                layer_result = layered_results[layer_name]
                flattened.extend(layer_result.results)
        
        return flattened
    
    def update_layered_search_config(self, new_config: Dict = None, **kwargs):
        """更新分层搜索配置"""
        if new_config:
            # 完全替换配置
            self.config.LAYERED_SEARCH_CONFIG = new_config
        else:
            # 部分更新配置
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    warnings.warn(f"Unknown config key: {key}")
        
        print(f"Layered search config updated")
        print(f"Current config: {self.config.LAYERED_SEARCH_CONFIG}")
    
    def get_layered_search_stats(self) -> Dict[str, Any]:
        """获取分层搜索统计信息"""
        stats = {
            'enabled': self.config.LAYERED_SEARCH_ENABLED,
            'fallback_enabled': self.config.LAYERED_SEARCH_FALLBACK,
            'deduplicate_enabled': self.config.LAYERED_SEARCH_DEDUPLICATE,
            'max_total_results': self.config.LAYERED_SEARCH_MAX_TOTAL_RESULTS,
            'layers': {}
        }
        
        # 添加各层配置详情
        for layer_name, layer_config in self.config.LAYERED_SEARCH_CONFIG.items():
            stats['layers'][layer_name] = {
                'similarity_range': layer_config.get('similarity_range', (0, 0)),
                'max_results': layer_config.get('max_results', 0),
                'heat_weight_factor': layer_config.get('heat_weight_factor', 1.0),
                'frequency_weight_factor': layer_config.get('frequency_weight_factor', 1.0),
                'recency_weight_factor': layer_config.get('recency_weight_factor', 1.0),
                'min_heat_required': layer_config.get('min_heat_required', 0)
            }
        
        return stats
    
    # =============== 簇内搜索方法 ===============
    
    def search_within_cluster(self, query_text: str = None, query_vector: np.ndarray = None, 
                             cluster_id: str = None, max_results: int = None) -> List[WeightedMemoryResult]:
        """
        在指定簇内搜索相似记忆，使用相对热力和访问频率加权
        
        参数:
            query_text: 查询文本（如果提供，将计算其嵌入向量）
            query_vector: 查询向量（如果提供，直接使用）
            cluster_id: 要搜索的簇ID（如果为None，则在整个记忆库中搜索）
            max_results: 最大返回结果数量
        
        返回:
            WeightedMemoryResult列表，按最终得分排序
        """
        if max_results is None:
            max_results = self.config.CLUSTER_SEARCH_MAX_RESULTS
        
        # 获取查询向量
        if query_vector is None and query_text is not None:
            query_vector = self._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")
        
        # 如果未指定簇ID，使用最相似的簇
        if cluster_id is None:
            # 寻找最相似的簇
            best_similarity = -1.0
            for cid, centroid in self.cluster_vectors.items():
                similarity = self._compute_similarity(query_vector, centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    cluster_id = cid
        
        if cluster_id not in self.clusters:
            return []
        
        # 检查缓存
        cached_results = self.cluster_search_cache.get(cluster_id, query_vector, self.current_turn)
        if cached_results is not None:
            self.stats['cache_hits'] += 1
            # 返回不超过max_results的结果
            return cached_results[:max_results]
        
        self.stats['cache_misses'] += 1
        self.stats['cluster_searches'] += 1
        
        cluster = self.clusters[cluster_id]
        cluster_total_heat = cluster.total_heat
        
        weighted_results = []
        
        # 获取簇内所有热区记忆
        cluster_memory_ids = set()
        
        # 方法1：从簇的memory_ids中获取（可能不完整）
        with cluster.lock:
            cluster_memory_ids.update(cluster.memory_ids)
        
        # 方法2：从hot_memories中筛选属于该簇的记忆
        for memory_id, memory in self.hot_memories.items():
            if memory.cluster_id == cluster_id:
                cluster_memory_ids.add(memory_id)
        
        # 计算加权得分
        for memory_id in cluster_memory_ids:
            memory = self.hot_memories.get(memory_id)
            if memory is None or memory.is_sleeping:
                # 如果记忆不在热区或是休眠状态，跳过
                continue
            
            # 计算基础相似度
            base_similarity = self._compute_similarity(query_vector, memory.vector)
            
            # 跳过相似度过低的记忆
            if base_similarity < self.config.SIMILARITY_THRESHOLD:
                continue
            
            # 计算各种权重
            relative_heat_weight = self._get_relative_heat_weight(memory, cluster_total_heat)
            access_frequency_weight = self._get_access_frequency_weight(memory_id, memory)
            recency_weight = self._get_recency_weight(memory)
            
            # 计算最终得分（加权几何平均）
            # 使用几何平均可以平衡各个权重的影响
            weights = [relative_heat_weight, access_frequency_weight, recency_weight]
            geometric_mean = np.exp(np.mean(np.log([max(0.0001, w) for w in weights])))
            
            final_score = base_similarity * geometric_mean
            
            # 创建加权结果对象
            result = WeightedMemoryResult(
                memory=memory,
                base_similarity=base_similarity,
                relative_heat_weight=relative_heat_weight,
                access_frequency_weight=access_frequency_weight,
                recency_weight=recency_weight,
                final_score=final_score,
                ranking_position=0  # 稍后设置
            )
            
            weighted_results.append(result)
        
        # 按最终得分排序
        weighted_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # 设置排名位置并限制结果数量
        for i, result in enumerate(weighted_results[:max_results]):
            result.ranking_position = i + 1
        
        final_results = weighted_results[:max_results]
        
        # 更新统计信息
        self.stats['weight_adjustments'] += len(final_results)
        
        # 检查是否有高频访问记忆
        high_freq_count = sum(1 for r in final_results 
                             if r.access_frequency_weight < 0.5)
        self.stats['high_frequency_memories'] += high_freq_count
        
        # 缓存结果（仅当结果非空时）
        if final_results:
            self.cluster_search_cache.put(cluster_id, query_vector, final_results, self.current_turn)
        
        return final_results
    
    def search_similar_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                               max_results: int = 10, use_weighting: bool = True) -> List[WeightedMemoryResult]:
        """
        搜索相似记忆（跨所有簇），可选择是否使用加权
        
        参数:
            query_text: 查询文本
            query_vector: 查询向量
            max_results: 最大返回结果数量
            use_weighting: 是否使用热力/频率加权
        
        返回:
            WeightedMemoryResult列表
        """
        if query_vector is None and query_text is not None:
            query_vector = self._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")
        
        all_results = []
        
        if use_weighting:
            # 使用加权搜索：首先在每个簇内搜索，然后合并结果
            for cluster_id in self.clusters.keys():
                cluster_results = self.search_within_cluster(
                    query_vector=query_vector,
                    cluster_id=cluster_id,
                    max_results=max_results // 2  # 从每个簇获取部分结果
                )
                all_results.extend(cluster_results)
        else:
            # 不使用加权：在整个记忆库中搜索
            for memory_id, memory in self.hot_memories.items():
                if memory.is_sleeping:
                    continue
                
                similarity = self._compute_similarity(query_vector, memory.vector)
                
                if similarity >= self.config.SIMILARITY_THRESHOLD:
                    result = WeightedMemoryResult(
                        memory=memory,
                        base_similarity=similarity,
                        relative_heat_weight=1.0,
                        access_frequency_weight=1.0,
                        recency_weight=1.0,
                        final_score=similarity,
                        ranking_position=0
                    )
                    all_results.append(result)
        
        # 排序并限制结果数量
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        
        for i, result in enumerate(all_results[:max_results]):
            result.ranking_position = i + 1
        
        return all_results[:max_results]
    
    def get_cluster_statistics(self, cluster_id: str) -> Dict[str, Any]:
        """获取簇的统计信息，包括热力分布、访问频率等"""
        if cluster_id not in self.clusters:
            return {}
        
        cluster = self.clusters[cluster_id]
        memories_in_cluster = []
        
        # 收集簇内记忆
        for memory_id, memory in self.hot_memories.items():
            if memory.cluster_id == cluster_id and not memory.is_sleeping:
                memories_in_cluster.append(memory)
        
        if not memories_in_cluster:
            return {
                'cluster_id': cluster_id,
                'total_heat': cluster.total_heat,
                'memory_count': 0,
                'heat_distribution': [],
                'frequency_stats': [],
                'current_turn': self.current_turn
            }
        
        # 计算热力分布
        heat_values = [m.heat for m in memories_in_cluster]
        total_heat = sum(heat_values)
        
        heat_distribution = []
        for memory in memories_in_cluster:
            if total_heat > 0:
                relative_heat = memory.heat / total_heat
            else:
                relative_heat = 0.0
            
            # 获取访问频率权重
            access_weight = self._get_access_frequency_weight(memory.id, memory)
            
            heat_distribution.append({
                'memory_id': memory.id,
                'heat': memory.heat,
                'relative_heat': relative_heat,
                'access_count': memory.access_count,
                'access_frequency_weight': access_weight,
                'last_interaction_turn': memory.last_interaction_turn,
                'turns_since_interaction': self.current_turn - memory.last_interaction_turn
            })
        
        # 按热力排序
        heat_distribution.sort(key=lambda x: x['heat'], reverse=True)
        
        # 计算访问频率统计
        frequency_stats = []
        for memory in memories_in_cluster:
            stats = self.access_frequency_stats.get(memory.id, {})
            frequency_stats.append({
                'memory_id': memory.id,
                'total_accesses': stats.get('count', memory.access_count),
                'recent_accesses': len(stats.get('recent_interactions', [])),
                'last_reset_turn': stats.get('last_reset_turn', memory.created_turn)
            })
        
        return {
            'cluster_id': cluster_id,
            'total_heat': cluster.total_heat,
            'memory_count': len(memories_in_cluster),
            'heat_distribution': heat_distribution[:10],  # 只返回前10个
            'frequency_stats': frequency_stats[:10],
            'average_heat': total_heat / len(memories_in_cluster) if memories_in_cluster else 0,
            'heat_std_dev': np.std(heat_values) if len(heat_values) > 1 else 0,
            'current_turn': self.current_turn,
            'cluster_last_updated_turn': cluster.last_updated_turn,
            'turns_since_cluster_update': self.current_turn - cluster.last_updated_turn
        }
    
    def reset_access_frequency(self, memory_id: str = None):
        """重置访问频率统计"""
        with self.frequency_stats_lock:
            if memory_id:
                if memory_id in self.access_frequency_stats:
                    self.access_frequency_stats[memory_id] = {
                        'count': 1,
                        'last_reset_turn': self.current_turn,
                        'recent_interactions': [self.current_turn]
                    }
            else:
                # 重置所有记忆的访问频率
                for mid in list(self.access_frequency_stats.keys()):
                    self.access_frequency_stats[mid] = {
                        'count': 1,
                        'last_reset_turn': self.current_turn,
                        'recent_interactions': [self.current_turn]
                    }
        
        # 使权重缓存失效
        self.weight_cache.clear()
    
    def adjust_memory_weights(self, memory_id: str, 
                             heat_adjustment: float = 1.0,
                             frequency_adjustment: float = 1.0):
        """手动调整记忆权重（用于特殊情况）"""
        if memory_id not in self.hot_memories:
            return False
        
        memory = self.hot_memories[memory_id]
        
        # 调整热力
        if heat_adjustment != 1.0:
            new_heat = int(memory.heat * heat_adjustment)
            with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
                tx.add_memory_heat_update(
                    memory_id,
                    memory.heat,
                    new_heat,
                    memory.cluster_id
                )
                memory.heat = new_heat
        
        # 调整访问频率权重（通过修改统计信息）
        if frequency_adjustment != 1.0:
            with self.frequency_stats_lock:
                if memory_id in self.access_frequency_stats:
                    stats = self.access_frequency_stats[memory_id]
                    # 通过调整访问计数来间接影响权重
                    adjusted_count = int(stats['count'] * frequency_adjustment)
                    stats['count'] = max(1, adjusted_count)
        
        # 清除相关缓存
        if memory.cluster_id:
            self.cluster_search_cache.clear(memory.cluster_id)
        
        # 使权重缓存失效
        if memory_id in self.weight_cache:
            del self.weight_cache[memory_id]
        
        return True
    
    # =============== 公共API（带缓存更新） ===============
    
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加新记忆（原子操作），包含重复检测和缓存更新"""
        # 增加轮数
        current_turn = self._increment_turn(self.config.TURN_INCREMENT_ON_ADD)
        
        # 计算嵌入
        vector = self._get_embedding(content)
        
        # 重复检测
        duplicate_id = self._check_duplicate(vector, content)
        if duplicate_id:
            # 如果是重复记忆，只更新访问记录，不分配热力
            with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
                memory = self.hot_memories.get(duplicate_id)
                if memory:
                    # 更新访问记录（使用轮数）
                    memory.last_interaction_turn = current_turn
                    memory.access_count += 1
                    memory.update_count += 1
                    
                    # 更新访问频率统计
                    self._update_access_frequency(duplicate_id)
                    
                    # 清除相关缓存
                    if memory.cluster_id:
                        self.cluster_search_cache.clear(memory.cluster_id)
                    
                    # 更新数据库
                    self.cursor.execute(f"""
                        UPDATE {self.config.MEMORY_TABLE}
                        SET last_interaction_turn = ?, access_count = ?, update_count = update_count + 1
                        WHERE id = ?
                    """, (memory.last_interaction_turn, memory.access_count, duplicate_id))
                    
                    # 更新统计
                    self.operation_count += 1
                    self.duplicate_skipped_count += 1
                    
                    # 事件驱动：检查是否触发维护任务
                    self._trigger_maintenance_if_needed()
                    
                    # 返回重复记忆的ID
                    return duplicate_id
            
            # 如果内存中没有，从数据库加载
            self.cursor.execute(f"SELECT * FROM {self.config.MEMORY_TABLE} WHERE id = ?", (duplicate_id,))
            row = self.cursor.fetchone()
            if row:
                # 只是更新访问记录
                self.cursor.execute(f"""
                    UPDATE {self.config.MEMORY_TABLE}
                    SET last_interaction_turn = ?, access_count = access_count + 1, update_count = update_count + 1
                    WHERE id = ?
                """, (current_turn, duplicate_id))
                self.duplicate_skipped_count += 1
                return duplicate_id
        
        # 使用事务确保原子性
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            # 检查热力池
            if self.heat_pool < self.config.NEW_MEMORY_HEAT:
                self._recycle_heat_pool()
            
            # 生成记忆ID
            memory_id = hashlib.md5(f"{content}_{current_turn}".encode()).hexdigest()[:16]
            
            # 创建记忆（使用轮数）
            memory = MemoryItem(
                id=memory_id,
                vector=vector,
                content=content,
                heat=0,
                created_turn=current_turn,
                last_interaction_turn=current_turn,
                metadata=metadata or {}
            )
            
            # 分配热力（应用抑制系数）
            base_allocated_heat = min(self.config.NEW_MEMORY_HEAT, self.heat_pool)
            suppression_factor = self._get_suppression_factor()
            
            if suppression_factor < 1.0:
                print(f"[Heat Suppression] Applying suppression factor {suppression_factor:.2f} to new memory")
                allocated_heat = int(base_allocated_heat * suppression_factor)
            else:
                allocated_heat = base_allocated_heat
            
            with self.heat_pool_lock:
                self.heat_pool -= allocated_heat
            
            # 寻找相似记忆
            neighbors = self._find_neighbors(vector, exclude_id=memory_id)
            
            if neighbors:
                # 应用抑制系数到邻居的热力分配
                neighbor_count = len(neighbors)
                if suppression_factor < 1.0:
                    allocation_per_neighbor = int((allocated_heat // (neighbor_count + 1)) * suppression_factor)
                else:
                    allocation_per_neighbor = allocated_heat // (neighbor_count + 1)
                
                total_allocated = 0
                for (neighbor_id, _, neighbor_memory) in neighbors:
                    new_heat = neighbor_memory.heat + allocation_per_neighbor
                    tx.add_memory_heat_update(
                        neighbor_id, 
                        neighbor_memory.heat, 
                        new_heat,
                        neighbor_memory.cluster_id
                    )
                    total_allocated += allocation_per_neighbor
                
                # 新记忆获得相同的热力
                memory.heat = allocation_per_neighbor
            else:
                memory.heat = allocated_heat
            
            # 分配到簇
            cluster_id = self._assign_to_cluster(memory, vector)
            
            # 调度簇质心更新
            self._schedule_cluster_centroid_update(cluster_id, vector, add=True)
            
            # 添加到热区
            self.hot_memories[memory_id] = memory
            self.memory_to_cluster[memory_id] = cluster_id
            
            # 初始化访问频率统计
            self._update_access_frequency(memory_id)
            
            # 记录记忆创建操作
            tx.add_memory_heat_update(memory_id, 0, memory.heat, cluster_id)
            
            # 更新统计
            self.stats['hot_memories'] += 1
            self.stats['total_memories'] += 1
            if suppression_factor < 1.0:
                self.stats['suppressed_memory_additions'] = self.stats.get('suppressed_memory_additions', 0) + 1
            
            # 更新事件计数器
            self.memory_addition_count += 1
            self.operation_count += 1
            
            # 事件驱动：检查是否触发维护任务
            self._trigger_maintenance_if_needed()
            
            # 保存到数据库
            self.cursor.execute(f"""
                INSERT INTO {self.config.MEMORY_TABLE} 
                (id, vector, content, heat, created_turn, last_interaction_turn, 
                 access_count, is_hot, is_sleeping, cluster_id, metadata, update_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                memory.id,
                self._vector_to_blob(memory.vector),
                memory.content,
                memory.heat,
                memory.created_turn,
                memory.last_interaction_turn,
                memory.access_count,
                int(memory.is_hot),
                int(memory.is_sleeping),
                memory.cluster_id,
                json.dumps(memory.metadata)
            ))
            
            # 更新向量缓存（增量）
            self.invalidate_vector_cache(memory_id, 'add')
            
            return memory_id
    
    def access_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """访问记忆"""
        # 增加轮数
        current_turn = self._increment_turn(self.config.TURN_INCREMENT_ON_ACCESS)
        
        # 在热区查找
        if memory_id in self.hot_memories:
            memory = self.hot_memories[memory_id]
            memory.last_interaction_turn = current_turn
            memory.access_count += 1
            memory.update_count += 1
            
            # 更新访问频率统计
            self._update_access_frequency(memory_id)
            
            # 清除相关缓存（因为记忆状态已改变）
            if memory.cluster_id:
                self.cluster_search_cache.clear(memory.cluster_id)
            
            # 更新事件计数器
            self.operation_count += 1
            
            # 事件驱动：检查是否触发维护任务
            self._trigger_maintenance_if_needed()
            
            # 更新数据库
            self.cursor.execute(f"""
                UPDATE {self.config.MEMORY_TABLE}
                SET last_interaction_turn = ?, access_count = ?, update_count = update_count + 1
                WHERE id = ?
            """, (memory.last_interaction_turn, memory.access_count, memory_id))
            
            # 使权重缓存失效
            if memory_id in self.weight_cache:
                del self.weight_cache[memory_id]
            
            return memory
        
        # 在冷区查找
        self.cursor.execute(f"SELECT * FROM {self.config.MEMORY_TABLE} WHERE id = ?", (memory_id,))
        row = self.cursor.fetchone()
        
        if row:
            # 解冻记忆
            with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
                # 更新热力池
                with self.heat_pool_lock:
                    if self.heat_pool >= self.config.INITIAL_HEAT_FOR_FROZEN:
                        self.heat_pool -= self.config.INITIAL_HEAT_FOR_FROZEN
                    else:
                        self._recycle_heat_pool()
                        if self.heat_pool >= self.config.INITIAL_HEAT_FOR_FROZEN:
                            self.heat_pool -= self.config.INITIAL_HEAT_FOR_FROZEN
                        else:
                            return None  # 热力池不足
                
                # 创建记忆对象
                memory = MemoryItem(
                    id=row['id'],
                    vector=self._blob_to_vector(row['vector']),
                    content=row['content'],
                    heat=self.config.INITIAL_HEAT_FOR_FROZEN,
                    created_turn=row['created_turn'],
                    last_interaction_turn=current_turn,
                    access_count=row['access_count'] + 1,
                    is_hot=True,
                    is_sleeping=False,
                    cluster_id=row['cluster_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    version=row['version'],
                    update_count=(row['update_count'] or 0) + 1
                )
                
                # 添加到热区
                self.hot_memories[memory_id] = memory
                self.memory_to_cluster[memory_id] = memory.cluster_id
                
                # 初始化访问频率统计
                self._update_access_frequency(memory_id)
                
                # 调度簇质心更新（记忆从冷区回到热区）
                if memory.cluster_id:
                    self._schedule_cluster_centroid_update(memory.cluster_id, memory.vector, add=True)
                    self.memory_additions_since_last_centroid_update += 1
                
                # 更新事件计数器
                self.memory_addition_count += 1
                self.operation_count += 1
                
                # 事件驱动：检查是否触发维护任务
                self._trigger_maintenance_if_needed()
                
                # 更新数据库
                self.cursor.execute(f"""
                    UPDATE {self.config.MEMORY_TABLE}
                    SET is_hot = 1, is_sleeping = 0, heat = ?, 
                        last_interaction_turn = ?, access_count = ?, update_count = ?
                    WHERE id = ?
                """, (memory.heat, memory.last_interaction_turn, memory.access_count, 
                      memory.update_count, memory_id))
                
                # 更新簇
                if memory.cluster_id and memory.cluster_id in self.clusters:
                    cluster = self.clusters[memory.cluster_id]
                    with cluster.lock:
                        cluster.hot_memory_count += 1
                        cluster.cold_memory_count -= 1
                        cluster.total_heat += memory.heat
                    
                    # 记录簇更新
                    tx.add_cluster_heat_update(memory.cluster_id, memory.heat)
                
                # 更新统计
                self.stats['hot_memories'] += 1
                self.stats['cold_memories'] -= 1
                
                # 更新向量缓存
                self.invalidate_vector_cache(memory_id, 'add')
                
                return memory
        
        return None
    
    def _recycle_heat_pool(self):
        """回收热力到热力池（原子操作）"""
        # 增加轮数
        self._increment_turn()
        
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            target_recycle = int(self.config.TOTAL_HEAT * 0.05)
            current_need = target_recycle - self.heat_pool
            
            if current_need <= 0:
                return
            
            # 收集符合条件的记忆
            eligible_memories = []
            for memory_id, memory in self.hot_memories.items():
                if memory.heat > 0 and not memory.is_sleeping:
                    eligible_memories.append((memory_id, memory))
            
            if not eligible_memories:
                return
            
            # 平均扣除热力
            deduct_per_memory = max(1, current_need // len(eligible_memories))
            total_recycled = 0
            
            for memory_id, memory in eligible_memories:
                if memory.heat > deduct_per_memory:
                    new_heat = memory.heat - deduct_per_memory
                    tx.add_memory_heat_update(memory_id, memory.heat, new_heat, memory.cluster_id)
                    
                    memory.heat = new_heat
                    memory.update_count += 1
                    total_recycled += deduct_per_memory
                else:
                    tx.add_memory_heat_update(memory_id, memory.heat, 0, memory.cluster_id)
                    
                    total_recycled += memory.heat
                    memory.heat = 0
                    memory.is_sleeping = True
                    memory.update_count += 1
                    self.sleeping_memories[memory_id] = memory
            
            # 更新热力池
            with self.heat_pool_lock:
                self.heat_pool += total_recycled
            
            # 更新事件计数器
            self.operation_count += len(eligible_memories)
            
            # 事件驱动：检查是否触发维护任务
            self._trigger_maintenance_if_needed()
            
            self.stats['total_heat_recycled'] += total_recycled
            self.stats['last_recycle_turn'] = self.current_turn
            
            # 检查休眠记忆
            if len(self.sleeping_memories) > 0:
                self._check_and_move_sleeping()
            
            # 使缓存失效
            self.weight_cache.clear()
            self.invalidate_vector_cache()
    
    # =============== 辅助方法 ===============
    
    def _find_neighbors(self, vector: np.ndarray, exclude_id: str = None, limit: int = None) -> List[Tuple[str, float, MemoryItem]]:
        """寻找相似邻居"""
        if limit is None:
            limit = self.config.MAX_NEIGHBORS
        
        neighbors = []
        for memory_id, memory in self.hot_memories.items():
            if exclude_id and memory_id == exclude_id:
                continue
            
            similarity = self._compute_similarity(vector, memory.vector)
            if similarity >= self.config.SIMILARITY_THRESHOLD:
                neighbors.append((memory_id, similarity, memory))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:limit]
    
    def _exponential_allocation(self, similarities: List[float], total_heat: int) -> List[int]:
        """指数函数分配热力（已弃用，改为平分）"""
        if not similarities:
            return []
        
        # 现在改为简单的平分
        neighbor_count = len(similarities)
        allocation_per_neighbor = total_heat // neighbor_count
        
        allocations = []
        total_allocated = 0
        
        for i in range(neighbor_count):
            if i == neighbor_count - 1:
                allocation = total_heat - total_allocated
            else:
                allocation = allocation_per_neighbor
                total_allocated += allocation
            
            allocations.append(allocation)
        
        return allocations
    
    def _assign_to_cluster(self, memory: MemoryItem, vector: np.ndarray) -> str:
        """分配到簇（原子操作）"""
        # 寻找最相似的簇
        best_cluster_id = None
        best_similarity = -1.0
        
        for cluster_id, centroid in self.cluster_vectors.items():
            similarity = self._compute_similarity(vector, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id
        
        if best_similarity >= self.config.CLUSTER_SIMILARITY_THRESHOLD:
            cluster_id = best_cluster_id
        else:
            # 创建新簇
            cluster_id = f"cluster_{self.current_turn}_{hashlib.md5(vector.tobytes()).hexdigest()[:8]}"
            cluster = SemanticCluster(
                id=cluster_id,
                centroid=vector.copy(),
                total_heat=0,
                hot_memory_count=0,
                cold_memory_count=0,
                is_loaded=True,
                size=0,
                last_updated_turn=self.current_turn,
                memory_additions_since_last_update=0
            )
            
            self.clusters[cluster_id] = cluster
            self.cluster_vectors[cluster_id] = cluster.centroid
            
            # 更新统计
            self.stats['clusters'] += 1
            
            # 保存到数据库
            self.cursor.execute(f"""
                INSERT INTO {self.config.CLUSTER_TABLE} 
                (id, centroid, total_heat, hot_memory_count, cold_memory_count, 
                 is_loaded, size, last_updated_turn, memory_additions_since_last_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cluster.id,
                self._vector_to_blob(cluster.centroid),
                cluster.total_heat,
                cluster.hot_memory_count,
                cluster.cold_memory_count,
                int(cluster.is_loaded),
                cluster.size,
                cluster.last_updated_turn,
                cluster.memory_additions_since_last_update
            ))
        
        # 更新簇信息
        cluster = self.clusters[cluster_id]
        with cluster.lock:
            cluster.memory_ids.add(memory.id)
            cluster.size += 1
            cluster.hot_memory_count += 1
            cluster.is_loaded = True
        
        memory.cluster_id = cluster_id
        return cluster_id
    
    def _check_and_move_sleeping(self):
        """检查并移动休眠记忆到冷区"""
        if len(self.sleeping_memories) == 0:
            return
        
        print(f"[Memory System] Moving {len(self.sleeping_memories)} sleeping memories to cold zone (Turn: {self.current_turn})")
        
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            for memory_id, memory in list(self.sleeping_memories.items()):
                # 移动到冷区
                memory.is_hot = False
                memory.is_sleeping = False
                memory.update_count += 1
                
                # 调度簇质心更新（记忆从热区移除）
                if memory.cluster_id:
                    self._schedule_cluster_centroid_update(memory.cluster_id, memory.vector, add=False)
                
                # 更新数据库
                self.cursor.execute(f"""
                    UPDATE {self.config.MEMORY_TABLE}
                    SET is_hot = 0, is_sleeping = 0, heat = 0, update_count = update_count + 1
                    WHERE id = ?
                """, (memory_id,))
                
                # 更新簇
                if memory.cluster_id and memory.cluster_id in self.clusters:
                    cluster = self.clusters[memory.cluster_id]
                    with cluster.lock:
                        cluster.hot_memory_count -= 1
                        cluster.cold_memory_count += 1
                        cluster.total_heat -= memory.heat
                    
                    # 记录簇更新
                    tx.add_cluster_heat_update(memory.cluster_id, -memory.heat)
                
                # 从热区移除
                del self.hot_memories[memory_id]
                del self.sleeping_memories[memory_id]
                
                # 从访问频率统计中移除
                with self.frequency_stats_lock:
                    if memory_id in self.access_frequency_stats:
                        del self.access_frequency_stats[memory_id]
                
                # 更新统计
                self.stats['hot_memories'] -= 1
                self.stats['cold_memories'] += 1
            
            # 更新向量缓存
            self.invalidate_vector_cache()
    
    def _vector_to_blob(self, vector: np.ndarray) -> bytes:
        """向量转换为二进制"""
        return vector.astype(np.float32).tobytes()
    
    def _blob_to_vector(self, blob: bytes) -> np.ndarray:
        """二进制转换为向量"""
        return np.frombuffer(blob, dtype=np.float32)
    
    def _update_cluster_in_db(self, cluster: SemanticCluster):
        """更新簇信息到数据库"""
        self.cursor.execute(f"""
            UPDATE {self.config.CLUSTER_TABLE}
            SET centroid = ?, total_heat = ?, hot_memory_count = ?, 
                cold_memory_count = ?, is_loaded = ?, size = ?, 
                last_updated_turn = ?, version = version + 1,
                memory_additions_since_last_update = ?
            WHERE id = ?
        """, (
            self._vector_to_blob(cluster.centroid),
            cluster.total_heat,
            cluster.hot_memory_count,
            cluster.cold_memory_count,
            int(cluster.is_loaded),
            cluster.size,
            self.current_turn,
            cluster.memory_additions_since_last_update,
            cluster.id
        ))
        cluster.version += 1
    
    # =============== 缓存统计和管理 ===============
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.vector_cache_lock:
            vector_cache_stats = {
                'is_valid': self.vector_cache.is_valid,
                'size': len(self.vector_cache.memory_ids) if self.vector_cache.memory_ids else 0,
                'last_updated': self.vector_cache.last_updated,
                'age_seconds': time.time() - self.vector_cache.last_updated if self.vector_cache.last_updated > 0 else 0
            }
        
        similarity_cache_stats = self.similarity_cache.get_stats()
        
        return {
            'vector_cache': vector_cache_stats,
            'similarity_cache': similarity_cache_stats,
            'weight_cache': {
                'size': len(self.weight_cache),
                'last_updated_turn': self.weight_cache_turn,
                'age_turns': self.current_turn - self.weight_cache_turn
            },
            'cluster_search_cache': {
                'size': len(self.cluster_search_cache.cache),
                'access_turns_size': len(self.cluster_search_cache.access_turns)
            },
            'memory_stats': {
                'hot_memories': len(self.hot_memories),
                'normalized_vectors_precomputed': self._normalized_vectors is not None,
                'memory_norms_precomputed': self._precomputed_memory_norms is not None
            }
        }
    
    def clear_all_caches(self):
        """清除所有缓存（用于调试）"""
        with self.vector_cache_lock:
            self.vector_cache.is_valid = False
            self.vector_cache.vectors = None
            self.vector_cache.memory_ids = None
        
        self.similarity_cache.cache.clear()
        self.weight_cache.clear()
        self.cluster_search_cache.clear()
        
        self._normalized_vectors = None
        self._precomputed_memory_norms = None
        
        print("[Cache] All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = self.stats.copy()
        
        # 新增热力分布统计
        cluster_heat_list = []
        total_cluster_heat = 0
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.total_heat > 0:
                cluster_heat_list.append({
                    'cluster_id': cluster_id,
                    'heat': cluster.total_heat,
                    'size': cluster.size
                })
                total_cluster_heat += cluster.total_heat
        
        if cluster_heat_list:
            cluster_heat_list.sort(key=lambda x: x['heat'], reverse=True)
            
            top3_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:min(3, len(cluster_heat_list))])
            top5_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:min(5, len(cluster_heat_list))])
            
            top3_ratio = top3_heat / total_cluster_heat if total_cluster_heat > 0 else 0
            top5_ratio = top5_heat / total_cluster_heat if total_cluster_heat > 0 else 0
            
            stats.update({
                'top3_heat_ratio': top3_ratio,
                'top5_heat_ratio': top5_ratio,
                'top3_exceeds_limit': top3_ratio > self.config.TOP3_HEAT_LIMIT_RATIO,
                'top5_exceeds_limit': top5_ratio > self.config.TOP5_HEAT_LIMIT_RATIO,
                'in_suppression_period': self._is_in_suppression_period(),
                'suppression_factor': self._get_suppression_factor(),
                'turns_since_last_recycle': self.current_turn - self.last_heat_recycle_turn if self.last_heat_recycle_turn > 0 else None,
                'heat_recycle_count': self.heat_recycle_count,
                'cluster_heat_history_size': sum(len(history) for history in self.cluster_heat_history.values()),
            })
        
        # 获取缓存统计
        cache_stats = self.get_cache_stats()
        
        stats.update({
            'memory_addition_count': self.memory_addition_count,
            'operation_count': self.operation_count,
            'memory_additions_since_last_centroid_update': self.memory_additions_since_last_centroid_update,
            'heat_pool': self.heat_pool,
            'hot_memories_count': len(self.hot_memories),
            'sleeping_memories_count': len(self.sleeping_memories),
            'clusters_count': len(self.clusters),
            'cluster_vectors_count': len(self.cluster_vectors),
            'pending_centroid_updates': sum(len(c.pending_centroid_updates) for c in self.clusters.values()),
            'update_queue_size': self.update_queue.qsize(),
            'clusters_needing_centroid_update': len(self.clusters_needing_centroid_update),
            'duplicate_detection_enabled': self.config.DUPLICATE_CHECK_ENABLED,
            'duplicate_threshold': self.config.DUPLICATE_THRESHOLD,
            'duplicate_skipped': self.duplicate_skipped_count,
            'current_turn': self.current_turn,
            'access_frequency_stats_size': len(self.access_frequency_stats),
            'cluster_search_cache_size': cache_stats['cluster_search_cache']['size'],
            'cache_hit_rate': (self.stats['cache_hits'] / 
                              max(1, self.stats['cache_hits'] + self.stats['cache_misses'])),
            'similarity_cache_hit_rate': (self.stats['similarity_cache_hits'] / 
                                         max(1, self.stats['similarity_cache_hits'] + self.stats['similarity_cache_misses'])),
            'weight_config': {
                'access_frequency_threshold': self.config.ACCESS_FREQUENCY_DISCOUNT_THRESHOLD,
                'access_frequency_discount_factor': self.config.ACCESS_FREQUENCY_DISCOUNT_FACTOR,
                'recency_decay_per_turn': self.config.RECENCY_WEIGHT_DECAY_PER_TURN,
                'relative_heat_power': self.config.RELATIVE_HEAT_WEIGHT_POWER
            },
            'heat_distribution_config': {
                'top3_limit': self.config.TOP3_HEAT_LIMIT_RATIO,
                'top5_limit': self.config.TOP5_HEAT_LIMIT_RATIO,
                'recycle_frequency': self.config.HEAT_RECYCLE_CHECK_FREQUENCY,
                'suppression_turns': self.config.HEAT_RECYCLE_SUPPRESSION_TURNS,
                'suppression_factor': self.config.HEAT_SUPPRESSION_FACTOR,
                'recycle_rate': self.config.HEAT_RECYCLE_RATE,
                'min_cluster_heat': self.config.MIN_CLUSTER_HEAT_AFTER_RECYCLE
            },
            'layered_search_config': {
                'enabled': self.config.LAYERED_SEARCH_ENABLED,
                'fallback': self.config.LAYERED_SEARCH_FALLBACK,
                'deduplicate': self.config.LAYERED_SEARCH_DEDUPLICATE,
                'max_total_results': self.config.LAYERED_SEARCH_MAX_TOTAL_RESULTS
            },
            'cache_stats': cache_stats
        })
        return stats
    
    def cleanup(self):
        """清理资源"""
        print(f"\n[Memory System] Cleaning up memory module (Final turn: {self.current_turn})...")
        
        # 执行一次最终维护任务
        self._perform_maintenance_tasks()
        
        # 创建最终检查点
        if self.memory_addition_count > 0:
            self._create_checkpoint()
        
        # 输出缓存统计
        cache_stats = self.get_cache_stats()
        print(f"[Cache] Final cache statistics:")
        print(f"  Vector cache: {cache_stats['vector_cache']['size']} vectors")
        print(f"  Similarity cache: {cache_stats['similarity_cache']['size']} entries")
        print(f"  Weight cache: {cache_stats['weight_cache']['size']} entries")
        print(f"  Cluster search cache: {cache_stats['cluster_search_cache']['size']} entries")
        
        # 关闭后台执行器
        self.background_executor.shutdown(wait=True)
        
        # 关闭数据库连接
        self.conn.close()
        
        print("[Memory System] Cleanup completed")
        
        # 输出最终统计
        print("\n[Memory System] Final statistics:")
        stats = self.get_stats()
        for key, value in stats.items():
            if isinstance(value, (int, float)) and value > 0:
                if key == 'cache_stats':
                    print(f"  {key}:")
                    for cat, cat_stats in value.items():
                        print(f"    {cat}:")
                        for k, v in cat_stats.items():
                            print(f"      {k}: {v}")
                else:
                    print(f"  {key}: {value}")


# =============== 示例用法 ===============
def example_layered_search():
    """示例：使用分层搜索"""
    # 初始化内存模块
    memory_module = MemoryModule()
    
    # 添加一些测试记忆
    test_memories = [
        "机器学习是人工智能的重要分支，用于模式识别和预测",
        "深度学习需要大量的计算资源，尤其是GPU",
        "自然语言处理让计算机理解人类语言，包括情感分析",
        "卷积神经网络在图像识别中表现出色，用于分类和检测",
        "Transformer模型彻底改变了自然语言处理的范式",
        "强化学习通过试错来学习最优策略，用于游戏和机器人",
        "生成对抗网络可以创造逼真的图像，用于艺术创作",
        "迁移学习利用已有知识解决新问题，提高学习效率",
        "自监督学习不需要人工标注的数据，降低成本",
        "联邦学习保护用户隐私的同时进行模型训练"
    ]
    
    print("Adding test memories...")
    for content in test_memories:
        memory_id = memory_module.add_memory(content)
        print(f"  Added: {content[:30]}...")
    
    # 测试分层搜索
    query = "人工智能和机器学习"
    
    print(f"\nSearching for: '{query}'")
    print("=" * 60)
    
    # 获取分层搜索结果
    layered_results = memory_module.search_layered_memories(query_text=query)
    
    # 显示各层结果
    total_results = 0
    for layer_name, layer_result in layered_results.items():
        sim_min, sim_max = layer_result.similarity_range
        print(f"\n{layer_name.upper()}: Similarity {sim_min:.2f}-{sim_max:.2f}")
        print(f"  Target: {layer_result.target_count}, Achieved: {layer_result.achieved_count}")
        print(f"  Avg Similarity: {layer_result.avg_similarity:.4f}, Avg Score: {layer_result.avg_final_score:.4f}")
        
        for i, result in enumerate(layer_result.results):
            print(f"    {i+1}. {result.memory.content[:40]}...")
            print(f"       Similarity: {result.base_similarity:.4f}, Final Score: {result.final_score:.4f}")
            print(f"       Heat: {result.memory.heat}, Access Count: {result.memory.access_count}")
        
        total_results += len(layer_result.results)
    
    print(f"\nTotal results across all layers: {total_results}")
    
    # 获取扁平化结果
    print(f"\nFlattened results:")
    flattened = memory_module.get_layered_search_results(query_text=query, flatten_results=True)
    for i, result in enumerate(flattened):
        print(f"{i+1}. [{result.memory.cluster_id[:8]}] {result.memory.content[:50]}...")
        print(f"   Similarity: {result.base_similarity:.4f}, Score: {result.final_score:.4f}")
    
    # 显示统计信息
    print(f"\nSystem statistics:")
    stats = memory_module.get_stats()
    print(f"Current turn: {stats['current_turn']}")
    print(f"Hot memories: {stats['hot_memories_count']}")
    print(f"Layered searches: {stats['layered_searches']}")
    print(f"Vectorized searches: {stats['vectorized_searches']}")
    print(f"Similarity cache hits: {stats['similarity_cache_hits']}, misses: {stats['similarity_cache_misses']}")
    
    # 清理
    memory_module.cleanup()


if __name__ == "__main__":
    print("="*80)
    print("Layered search example with NumPy vectorization:")
    print("="*80)
    example_layered_search()
