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
    
    # 指数分配参数
    EXPONENT_BASE = 2.0  # 指数分配的底数
    MIN_ALLOCATION_RATIO = 0.01  # 最小分配比例
    
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
        self.started_at = time.time()
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
    
    def add_memory_heat_update(self, memory_id: str, old_heat: int, new_heat: int, cluster_id: str = None):
        """添加记忆热力更新操作"""
        self.operations.append({
            'type': OperationType.MEMORY_HEAT_UPDATE,
            'memory_id': memory_id,
            'old_heat': old_heat,
            'new_heat': new_heat,
            'cluster_id': cluster_id,
            'timestamp': time.time()
        })
        
        # 记录热力变化
        heat_delta = new_heat - old_heat
        if cluster_id and heat_delta != 0:
            if cluster_id not in self.cluster_updates:
                self.cluster_updates[cluster_id] = 0
            self.cluster_updates[cluster_id] += heat_delta
    
    def add_cluster_heat_update(self, cluster_id: str, heat_delta: int):
        """添加簇热力更新操作"""
        self.operations.append({
            'type': OperationType.CLUSTER_HEAT_UPDATE,
            'cluster_id': cluster_id,
            'heat_delta': heat_delta,
            'timestamp': time.time()
        })

# =============== 数据类定义 ===============
@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    vector: np.ndarray
    content: str
    heat: int = 0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    is_hot: bool = True
    is_sleeping: bool = False
    cluster_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1  # 版本号，用于乐观锁
    pending_updates: List[Dict] = field(default_factory=list)  # 待应用的更新
    
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
    last_updated: float = field(default_factory=time.time)
    version: int = 1
    lock: threading.RLock = field(default_factory=threading.RLock)  # 每个簇有自己的锁
    pending_heat_delta: int = 0  # 待应用的热力变化
    pending_centroid_updates: List[Tuple[np.ndarray, bool]] = field(default_factory=list)  # (向量, add=True/remove=False)
    memory_additions_since_last_update: int = 0  # 上次更新后新增的记忆数

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
    """内存管理模块 - 纯事件驱动设计"""
    
    def __init__(self, embedding_func=None, similarity_func=None):
        self.config = Config()
        
        # 纯事件驱动配置（完全移除定时器）
        self.CHECKPOINT_MEMORY_THRESHOLD = 100  # 每添加100次记忆创建检查点
        self.CONSISTENCY_CHECK_THRESHOLD = 50   # 每50次操作检查一致性
        self.MAINTENANCE_OPERATION_THRESHOLD = 200  # 每200次操作执行完整维护
        
        # 事件计数器
        self.memory_addition_count = 0
        self.operation_count = 0
        self.memory_additions_since_last_centroid_update: int = 0
        
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
        
        # 统计信息
        self.stats = {
            'total_memories': 0,
            'hot_memories': 0,
            'cold_memories': 0,
            'clusters': 0,
            'loaded_clusters': 0,
            'total_heat_recycled': 0,
            'total_heat_allocated': 0,
            'last_recycle_time': 0,
            'consistency_violations': 0,
            'transaction_retries': 0,
            'centroid_updates': 0,
            'full_centroid_recalculations': 0,
            'maintenance_cycles': 0,
            'events_triggered': 0
        }
        
        # 加载系统状态
        self._load_system_state()
        
        print(f"Memory Module initialized with PURE event-driven design")
        print(f"No background maintenance threads - all maintenance triggered by events")
        print(f"Embedding model: {'External' if self._external_embedding_func else 'Internal'}")
        print(f"Checkpoint threshold: {self.CHECKPOINT_MEMORY_THRESHOLD} memory additions")
        print(f"Consistency threshold: {self.CONSISTENCY_CHECK_THRESHOLD} operations")
        print(f"Maintenance threshold: {self.MAINTENANCE_OPERATION_THRESHOLD} total operations")
        print(f"Duplicate detection: {'Enabled' if self.config.DUPLICATE_CHECK_ENABLED else 'Disabled'} "
              f"(threshold: {self.config.DUPLICATE_THRESHOLD})")
    
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
        """初始化数据库表结构"""
        self.conn = sqlite3.connect(self.config.DB_PATH, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式提高并发
        self.conn.execute("PRAGMA synchronous=NORMAL")  # 平衡性能和数据安全
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # 创建统一记忆表
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.MEMORY_TABLE} (
                id TEXT PRIMARY KEY,
                vector BLOB,
                content TEXT NOT NULL,
                heat INTEGER DEFAULT 0,
                last_accessed REAL,
                access_count INTEGER DEFAULT 1,
                is_hot INTEGER DEFAULT 1,
                is_sleeping INTEGER DEFAULT 0,
                cluster_id TEXT,
                created_at REAL,
                metadata TEXT,
                version INTEGER DEFAULT 1,
                last_updated REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # 创建簇表
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.CLUSTER_TABLE} (
                id TEXT PRIMARY KEY,
                centroid BLOB,
                total_heat INTEGER DEFAULT 0,
                hot_memory_count INTEGER DEFAULT 0,
                cold_memory_count INTEGER DEFAULT 0,
                is_loaded INTEGER DEFAULT 0,
                size INTEGER DEFAULT 0,
                last_updated REAL,
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
                version INTEGER DEFAULT 1
            )
        """)
        
        # 创建操作日志表
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.OPERATION_LOG_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                operation_type TEXT,
                memory_id TEXT,
                cluster_id TEXT,
                old_value TEXT,
                new_value TEXT,
                timestamp REAL,
                applied INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0
            )
        """)
        
        # 创建索引
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_op_log_timestamp ON {self.config.OPERATION_LOG_TABLE}(timestamp)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_op_log_applied ON {self.config.OPERATION_LOG_TABLE}(applied)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_cluster ON {self.config.MEMORY_TABLE}(cluster_id, heat)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_hot_heat ON {self.config.MEMORY_TABLE}(is_hot, heat DESC)")
        
        # 初始化表
        self.cursor.execute(f"""
            INSERT OR IGNORE INTO {self.config.HEAT_POOL_TABLE} (id, heat_pool, total_allocated_heat)
            VALUES (1, {self.config.INITIAL_HEAT_POOL}, 0)
        """)
        
        self.conn.commit()
    
    def _load_system_state(self):
        """加载系统状态"""
        # 加载热力池
        self.cursor.execute(f"SELECT heat_pool, total_allocated_heat FROM {self.config.HEAT_POOL_TABLE} WHERE id = 1")
        row = self.cursor.fetchone()
        if row:
            self.heat_pool = row['heat_pool']
            self.total_allocated_heat = row['total_allocated_heat']
        
        # 加载所有簇
        self._load_all_clusters()
        
        # 加载热区记忆
        self._load_hot_memories()
        
        # 初始一致性检查
        self._check_consistency()
    
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
                last_updated=row['last_updated'],
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
                last_accessed=row['last_accessed'],
                access_count=row['access_count'],
                is_hot=bool(row['is_hot']),
                is_sleeping=bool(row['is_sleeping']),
                cluster_id=row['cluster_id'],
                created_at=row['created_at'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                version=row['version']
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
                'timestamp': time.time()
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
                    'timestamp': time.time()
                })
            else:
                self.conn.rollback()
                self.operation_log.append({
                    'transaction_id': transaction_id,
                    'type': 'rollback',
                    'timestamp': time.time()
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
            # 提交维护任务到线程池
            self.background_executor.submit(self._perform_maintenance_tasks)
            
            # 重置计数器（避免重复触发）
            if self.operation_count >= self.MAINTENANCE_OPERATION_THRESHOLD:
                self.operation_count = 0
            if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
                self.memory_addition_count = 0
    
    def _perform_maintenance_tasks(self):
        """执行维护任务 - 完全事件驱动"""
        print(f"\n[Memory System] Performing maintenance tasks")
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
        
        # 5. 检查是否需要创建检查点
        self._create_checkpoint_if_needed()
        
        # 6. 检查并处理休眠记忆
        if len(self.sleeping_memories) > 0:
            self._check_and_move_sleeping()
        
        # 7. 更新内存缓存状态
        self._update_memory_cache_state()
        
        elapsed = time.time() - start_time
        self.stats['maintenance_cycles'] += 1
        
        print(f"[Memory System] Maintenance cycle {self.stats['maintenance_cycles']} completed in {elapsed:.2f}s")
    
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
        print(f"[Memory System] Creating system checkpoint")
        
        try:
            # 保存所有热区记忆
            for memory in self.hot_memories.values():
                self.cursor.execute(f"""
                    UPDATE {self.config.MEMORY_TABLE}
                    SET heat = ?, last_accessed = ?, access_count = ?, 
                        is_hot = ?, is_sleeping = ?, version = version + 1
                    WHERE id = ?
                """, (
                    memory.heat,
                    memory.last_accessed,
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
                        size = ?, last_updated = ?, version = version + 1,
                        memory_additions_since_last_update = ?
                    WHERE id = ?
                """, (
                    self._vector_to_blob(cluster.centroid),
                    cluster.total_heat,
                    cluster.hot_memory_count,
                    cluster.cold_memory_count,
                    cluster.size,
                    time.time(),
                    cluster.memory_additions_since_last_update,
                    cluster.id
                ))
            
            # 保存热力池
            with self.heat_pool_lock:
                self.cursor.execute(f"""
                    UPDATE {self.config.HEAT_POOL_TABLE}
                    SET heat_pool = ?, total_allocated_heat = ?, version = version + 1
                    WHERE id = 1
                """, (self.heat_pool, self.total_allocated_heat))
            
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
            'timestamp': time.time()
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
                'timestamp': time.time(),
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
            'timestamp': time.time()
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
            
            # 更新数据库
            if immediate:
                self.cursor.execute(f"""
                    UPDATE {self.config.MEMORY_TABLE}
                    SET heat = ?, version = version + 1, last_updated = ?
                    WHERE id = ?
                """, (new_heat, time.time(), memory_id))
            
            # 更新簇热力（如果需要）
            if cluster_id and cluster_id in self.clusters:
                heat_delta = new_heat - operation['old_heat']
                if heat_delta != 0:
                    self._update_cluster_heat(cluster_id, heat_delta, immediate)
    
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
    
    def _log_operation(self, operation: Dict, applied: bool = False):
        """记录操作到日志"""
        self.cursor.execute(f"""
            INSERT INTO {self.config.OPERATION_LOG_TABLE}
            (transaction_id, operation_type, memory_id, cluster_id, 
             old_value, new_value, timestamp, applied)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            operation.get('transaction_id'),
            operation['type'].value,
            operation.get('memory_id'),
            operation.get('cluster_id'),
            json.dumps(operation.get('old_value', {})),
            json.dumps(operation.get('new_value', {})),
            time.time(),
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
                'timestamp': time.time()
            })
    
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
                    'timestamp': item['timestamp']
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
                        SET centroid = ?, last_updated = ?, version = version + 1,
                            memory_additions_since_last_update = 0
                        WHERE id = ?
                    """, (
                        self._vector_to_blob(update_data['centroid']),
                        update_data['timestamp'],
                        cluster_id
                    ))
        
        # 批量更新记忆
        for memory_id, new_heat in memory_updates.items():
            if memory_id in self.hot_memories:
                self.hot_memories[memory_id].heat = new_heat
    
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
                    cluster.last_updated = time.time()
                    cluster.version += 1
                    
                    # 更新向量缓存
                    self.cluster_vectors[cluster_id] = new_centroid
                    
                    # 记录更新
                    centroid_updates[cluster_id] = {
                        'centroid': new_centroid,
                        'timestamp': time.time()
                    }
        
        # 批量更新数据库
        if centroid_updates:
            with self.conn:
                for cluster_id, update_data in centroid_updates.items():
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET centroid = ?, last_updated = ?, version = version + 1,
                            memory_additions_since_last_update = 0
                        WHERE id = ?
                    """, (
                        self._vector_to_blob(update_data['centroid']),
                        update_data['timestamp'],
                        cluster_id
                    ))
            
            self.stats['centroid_updates'] += len(centroid_updates)
            print(f"[Memory System] Updated centroids for {len(centroid_updates)} clusters")
    
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
    
    # =============== 公共API ===============
    
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加新记忆（原子操作），包含重复检测"""
        # 计算嵌入
        vector = self._get_embedding(content)
        
        # 重复检测
        duplicate_id = self._check_duplicate(vector, content)
        if duplicate_id:
            # 如果是重复记忆，只更新访问记录，不分配热力
            with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
                memory = self.hot_memories.get(duplicate_id)
                if memory:
                    # 更新访问记录
                    memory.last_accessed = time.time()
                    memory.access_count += 1
                    
                    # 更新数据库
                    self.cursor.execute(f"""
                        UPDATE {self.config.MEMORY_TABLE}
                        SET last_accessed = ?, access_count = ?
                        WHERE id = ?
                    """, (memory.last_accessed, memory.access_count, duplicate_id))
                    
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
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (time.time(), duplicate_id))
                self.duplicate_skipped_count += 1
                return duplicate_id
        
        # 使用事务确保原子性
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            # 检查热力池
            if self.heat_pool < self.config.NEW_MEMORY_HEAT:
                self._recycle_heat_pool()
            
            # 生成记忆ID
            memory_id = hashlib.md5(f"{content}_{time.time()}".encode()).hexdigest()[:16]
            
            # 创建记忆
            memory = MemoryItem(
                id=memory_id,
                vector=vector,
                content=content,
                heat=0,
                metadata=metadata or {}
            )
            
            # 分配热力（原子操作）
            allocated_heat = min(self.config.NEW_MEMORY_HEAT, self.heat_pool)
            with self.heat_pool_lock:
                self.heat_pool -= allocated_heat
            
            # 寻找相似记忆
            neighbors = self._find_neighbors(vector, exclude_id=memory_id)
            
            if neighbors:
                # 分配热力给邻居
                similarities = [sim for _, sim, _ in neighbors]
                allocations = self._exponential_allocation(similarities, allocated_heat // 2)
                
                total_allocated = 0
                for (neighbor_id, _, neighbor_memory), allocation in zip(neighbors, allocations):
                    new_heat = neighbor_memory.heat + allocation
                    tx.add_memory_heat_update(
                        neighbor_id, 
                        neighbor_memory.heat, 
                        new_heat,
                        neighbor_memory.cluster_id
                    )
                    total_allocated += allocation
                
                # 剩余热力给新记忆
                memory.heat = allocated_heat - total_allocated
            else:
                memory.heat = allocated_heat
            
            # 分配到簇
            cluster_id = self._assign_to_cluster(memory, vector)
            
            # 调度簇质心更新
            self._schedule_cluster_centroid_update(cluster_id, vector, add=True)
            
            # 添加到热区
            self.hot_memories[memory_id] = memory
            self.memory_to_cluster[memory_id] = cluster_id
            
            # 记录记忆创建操作
            tx.add_memory_heat_update(memory_id, 0, memory.heat, cluster_id)
            
            # 更新统计
            self.stats['hot_memories'] += 1
            self.stats['total_memories'] += 1
            
            # 更新事件计数器
            self.memory_addition_count += 1
            self.operation_count += 1
            
            # 事件驱动：检查是否触发维护任务
            self._trigger_maintenance_if_needed()
            
            # 保存到数据库
            self.cursor.execute(f"""
                INSERT INTO {self.config.MEMORY_TABLE} 
                (id, vector, content, heat, last_accessed, access_count, 
                 is_hot, is_sleeping, cluster_id, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                self._vector_to_blob(memory.vector),
                memory.content,
                memory.heat,
                memory.last_accessed,
                memory.access_count,
                int(memory.is_hot),
                int(memory.is_sleeping),
                memory.cluster_id,
                memory.created_at,
                json.dumps(memory.metadata)
            ))
            
            return memory_id
    
    def access_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """访问记忆"""
        # 在热区查找
        if memory_id in self.hot_memories:
            memory = self.hot_memories[memory_id]
            memory.last_accessed = time.time()
            memory.access_count += 1
            
            # 更新事件计数器
            self.operation_count += 1
            
            # 事件驱动：检查是否触发维护任务
            self._trigger_maintenance_if_needed()
            
            # 更新数据库
            self.cursor.execute(f"""
                UPDATE {self.config.MEMORY_TABLE}
                SET last_accessed = ?, access_count = ?
                WHERE id = ?
            """, (memory.last_accessed, memory.access_count, memory_id))
            
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
                    last_accessed=time.time(),
                    access_count=row['access_count'] + 1,
                    is_hot=True,
                    cluster_id=row['cluster_id'],
                    created_at=row['created_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                
                # 添加到热区
                self.hot_memories[memory_id] = memory
                self.memory_to_cluster[memory_id] = memory.cluster_id
                
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
                    SET is_hot = 1, heat = ?, last_accessed = ?, access_count = ?
                    WHERE id = ?
                """, (memory.heat, memory.last_accessed, memory.access_count, memory_id))
                
                # 更新簇
                if memory.cluster_id and memory.cluster_id in self.clusters:
                    cluster = self.clusters[memory.cluster_id]
                    cluster.hot_memory_count += 1
                    cluster.cold_memory_count -= 1
                    cluster.total_heat += memory.heat
                    
                    # 记录簇更新
                    tx.add_cluster_heat_update(memory.cluster_id, memory.heat)
                
                # 更新统计
                self.stats['hot_memories'] += 1
                self.stats['cold_memories'] -= 1
                
                return memory
        
        return None
    
    def _recycle_heat_pool(self):
        """回收热力到热力池（原子操作）"""
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
                    total_recycled += deduct_per_memory
                else:
                    tx.add_memory_heat_update(memory_id, memory.heat, 0, memory.cluster_id)
                    
                    total_recycled += memory.heat
                    memory.heat = 0
                    memory.is_sleeping = True
                    self.sleeping_memories[memory_id] = memory
            
            # 更新热力池
            with self.heat_pool_lock:
                self.heat_pool += total_recycled
            
            # 更新事件计数器
            self.operation_count += len(eligible_memories)
            
            # 事件驱动：检查是否触发维护任务
            self._trigger_maintenance_if_needed()
            
            self.stats['total_heat_recycled'] += total_recycled
            
            # 检查休眠记忆
            if len(self.sleeping_memories) > 0:
                self._check_and_move_sleeping()
    
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
        """指数函数分配热力"""
        if not similarities:
            return []
        
        weights = [self.config.EXPONENT_BASE ** (sim * 10) for sim in similarities]
        total_weight = sum(weights)
        
        min_weight = total_weight * self.config.MIN_ALLOCATION_RATIO
        weights = [max(w, min_weight) for w in weights]
        total_weight = sum(weights)
        
        allocations = []
        total_allocated = 0
        
        for i, weight in enumerate(weights):
            if i == len(weights) - 1:
                allocation = total_heat - total_allocated
            else:
                allocation = int(total_heat * weight / total_weight)
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
            cluster_id = f"cluster_{int(time.time())}_{hashlib.md5(vector.tobytes()).hexdigest()[:8]}"
            cluster = SemanticCluster(
                id=cluster_id,
                centroid=vector.copy(),
                total_heat=0,
                hot_memory_count=0,
                cold_memory_count=0,
                is_loaded=True,
                size=0,
                last_updated=time.time(),
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
                 is_loaded, size, last_updated, memory_additions_since_last_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cluster.id,
                self._vector_to_blob(cluster.centroid),
                cluster.total_heat,
                cluster.hot_memory_count,
                cluster.cold_memory_count,
                int(cluster.is_loaded),
                cluster.size,
                cluster.last_updated,
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
        
        print(f"[Memory System] Moving {len(self.sleeping_memories)} sleeping memories to cold zone")
        
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            for memory_id, memory in list(self.sleeping_memories.items()):
                # 移动到冷区
                memory.is_hot = False
                memory.is_sleeping = False
                
                # 调度簇质心更新（记忆从热区移除）
                if memory.cluster_id:
                    self._schedule_cluster_centroid_update(memory.cluster_id, memory.vector, add=False)
                
                # 更新数据库
                self.cursor.execute(f"""
                    UPDATE {self.config.MEMORY_TABLE}
                    SET is_hot = 0, is_sleeping = 0, heat = 0
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
                
                # 更新统计
                self.stats['hot_memories'] -= 1
                self.stats['cold_memories'] += 1
    
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
                last_updated = ?, version = version + 1,
                memory_additions_since_last_update = ?
            WHERE id = ?
        """, (
            self._vector_to_blob(cluster.centroid),
            cluster.total_heat,
            cluster.hot_memory_count,
            cluster.cold_memory_count,
            int(cluster.is_loaded),
            cluster.size,
            time.time(),
            cluster.memory_additions_since_last_update,
            cluster.id
        ))
        cluster.version += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = self.stats.copy()
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
        })
        return stats
    
    def cleanup(self):
        """清理资源"""
        print("\n[Memory System] Cleaning up memory module...")
        
        # 执行一次最终维护任务
        self._perform_maintenance_tasks()
        
        # 创建最终检查点
        if self.memory_addition_count > 0:
            self._create_checkpoint()
        
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
                print(f"  {key}: {value}")


def example_usage():
    """示例用法"""
    # 初始化内存模块（使用外部嵌入函数）
    def custom_embedding(text: str) -> np.ndarray:
        # 这里可以使用任何嵌入模型
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text, show_progress_bar=False)
    
    def custom_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    # 使用外部嵌入函数初始化
    memory_module = MemoryModule(
        embedding_func=custom_embedding,
        similarity_func=custom_similarity
    )
    
    # 添加记忆
    memory_id = memory_module.add_memory("今天天气真好，适合出去散步")
    print(f"Added memory: {memory_id}")
    
    # 访问记忆
    memory = memory_module.access_memory(memory_id)
    if memory:
        print(f"Accessed memory: {memory.content}")
    
    # 获取统计
    stats = memory_module.get_stats()
    print(f"Hot memories: {stats['hot_memories_count']}")
    
    # 清理
    memory_module.cleanup()


if __name__ == "__main__":
    example_usage()