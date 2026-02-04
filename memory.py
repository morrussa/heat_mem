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
from typing import List, Tuple, Optional, Any, Dict, Set
# =============== 尝试导入Annoy ===============
try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    print("Warning: Annoy not installed. Using fallback clustering.")

# =============== 工具函数（独立于类） ===============

# 1. 向量与二进制转换
def vector_to_blob(vector: np.ndarray) -> bytes:
    """向量转换为二进制"""
    return vector.astype(np.float32).tobytes()

def blob_to_vector(blob: bytes) -> np.ndarray:
    """二进制转换为向量"""
    return np.frombuffer(blob, dtype=np.float32)

# 2. 轮数递增
def increment_turn(current_turn: int, stats: Dict, turn_lock: threading.RLock, 
                   increment: int = 1) -> int:
    """统一轮数递增，返回新的轮数值"""
    with turn_lock:
        current_turn += increment
        stats['current_turn'] = current_turn
    return current_turn

# 3. 热力池操作
def allocate_heat_from_pool(heat_pool: int, needed_heat: int, 
                           total_allocated_heat: int, 
                           heat_pool_lock: threading.RLock) -> Tuple[int, int]:
    """从热力池分配热力"""
    with heat_pool_lock:
        actual_allocated = min(needed_heat, heat_pool)
        heat_pool -= actual_allocated
        total_allocated_heat += actual_allocated
    return heat_pool, actual_allocated

def update_memory_heat_in_db(cursor, table_name: str, memory_id: str, new_heat: int, 
                            update_count_increment: int = 1):
    """更新记忆热力到数据库"""
    cursor.execute(f"""
        UPDATE {table_name}
        SET heat = ?, update_count = update_count + ?
        WHERE id = ?
    """, (new_heat, update_count_increment, memory_id))

def update_cluster_heat_in_db(cursor, table_name: str, cluster_id: str, heat_delta: int):
    """更新簇热力到数据库"""
    if heat_delta != 0:
        cursor.execute(f"""
            UPDATE {table_name}
            SET total_heat = total_heat + ?, version = version + 1
            WHERE id = ?
        """, (heat_delta, cluster_id))

# 4. 缓存管理
def invalidate_memory_caches(caches_dict: Dict, memory_id: str = None, 
                            cluster_id: str = None, full: bool = False):
    """统一缓存失效管理"""
    if full:
        caches_dict.clear()
        return
    
    if memory_id and memory_id in caches_dict:
        del caches_dict[memory_id]
    
    if cluster_id:
        keys_to_remove = [k for k in caches_dict.keys() 
                         if k.startswith(f'cluster_{cluster_id}_')]
        for key in keys_to_remove:
            del caches_dict[key]

# 5. 簇质心调度
def schedule_centroid_update(clusters: Dict[str, 'SemanticCluster'], 
                           cluster_id: str, vector: np.ndarray, 
                           clusters_needing_update: Set[str], add: bool = True):
    """调度簇质心更新"""
    if cluster_id not in clusters:
        return
    
    cluster = clusters[cluster_id]
    with cluster.lock:
        cluster.pending_centroid_updates.append((vector.copy(), add))
        if add:
            cluster.memory_additions_since_last_update += 1
        clusters_needing_update.add(cluster_id)

# 6. 事务操作
def execute_with_retry(func, max_retries: int = 3, 
                       retry_delay: float = 0.1, *args, **kwargs):
    """带重试的执行函数"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (2 ** attempt))
            continue

# 7. 向量相似度计算
def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

# 8. 批量向量相似度计算
def compute_batch_similarities(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """批量计算相似度"""
    if vectors.shape[0] == 0:
        return np.array([])
    
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros(vectors.shape[0])
    
    normalized_query = query_vector / query_norm
    memory_norms = np.linalg.norm(vectors, axis=1)
    memory_norms[memory_norms == 0] = 1e-10
    normalized_vectors = vectors / memory_norms[:, np.newaxis]
    
    similarities = np.dot(normalized_vectors, normalized_query)
    return np.clip(similarities, -1.0, 1.0)

# 9. 内存向量转换
def convert_memory_vectors(memories: List['MemoryItem']) -> Tuple[List[str], np.ndarray]:
    """将记忆列表转换为ID列表和向量数组"""
    memory_ids = []
    vectors = []
    
    for memory in memories:
        memory_ids.append(memory.id)
        vectors.append(memory.vector)
    
    if vectors:
        vectors_array = np.array(vectors, dtype=np.float32)
    else:
        vectors_array = np.zeros((0, 1024), dtype=np.float32)
    
    return memory_ids, vectors_array

# =============== 配置常量 ===============
class Config:
    # 热力系统配置
    TOTAL_HEAT = 10000000
    HEAT_POOL_RATIO = 0.1
    INITIAL_HEAT_POOL = int(TOTAL_HEAT * HEAT_POOL_RATIO)
    HEAT_POOL_RECYCLE_THRESHOLD = 0.2  # 热力池低于20%时触发回收
    
    # 新记忆分配配置
    NEW_MEMORY_HEAT = 50000
    SIMILARITY_THRESHOLD = 0.75
    MAX_NEIGHBORS = 5
    
    # 重复检测配置
    DUPLICATE_THRESHOLD = 0.95
    DUPLICATE_CHECK_ENABLED = True
    
    # 热区管理配置
    HOT_ZONE_RATIO = 0.2
    SINGLE_MEMORY_LIMIT_RATIO = 0.05
    SLEEP_MEMORY_SIZE_LIMIT = 512 * 1024 * 1024
    
    # 冷区管理配置
    DELAYED_UPDATE_LIMIT = 10
    INITIAL_HEAT_FOR_FROZEN = 1000
    
    # 语义簇配置
    CLUSTER_SIMILARITY_THRESHOLD = 0.85
    CLUSTER_MIN_SIZE = 3
    CLUSTER_MAX_SIZE = 1000
    CLUSTER_MERGE_THRESHOLD = 0.90
    CLUSTER_SPLIT_THRESHOLD = 100
    
    # 质心更新配置
    CENTROID_UPDATE_FREQUENCY = 10
    CENTROID_UPDATE_BATCH_SIZE = 100
    CENTROID_FULL_RECALC_THRESHOLD = 1000
    
    # 模型配置
    MODEL_PATH = None
    EMBEDDING_DIM = 1024
    
    # 数据库配置
    DB_PATH = "./memory/memory.db"
    MEMORY_TABLE = "memories"
    CLUSTER_TABLE = "clusters"
    HEAT_POOL_TABLE = "heat_pool"
    STATS_TABLE = "system_stats"
    OPERATION_LOG_TABLE = "operation_log"
    
    # 系统参数
    BACKGROUND_THREADS = 4
    CLUSTER_LOAD_BATCH_SIZE = 50
    OPERATION_BATCH_SIZE = 100
    
    # 事务配置
    TRANSACTION_TIMEOUT = 30
    MAX_RETRY_COUNT = 3
    
    # 锁配置
    MEMORY_LOCK_TIMEOUT = 5
    CLUSTER_LOCK_TIMEOUT = 3
    
    # 基于轮数的时间系统配置
    INITIAL_TURN = 0
    TURN_INCREMENT_ON_ACCESS = 1
    TURN_INCREMENT_ON_ADD = 1
    TURN_INCREMENT_ON_MAINTENANCE = 1
    
    # 簇内搜索配置
    CLUSTER_SEARCH_MAX_RESULTS = 20
    ACCESS_FREQUENCY_DISCOUNT_THRESHOLD = 50
    ACCESS_FREQUENCY_DISCOUNT_FACTOR = 0.7
    RECENCY_WEIGHT_DECAY_PER_TURN = 0.001
    RELATIVE_HEAT_WEIGHT_POWER = 0.5
    
    # 簇内搜索缓存配置
    CLUSTER_SEARCH_CACHE_SIZE = 50
    CLUSTER_SEARCH_CACHE_TTL_TURNS = 100
    
    # 热力分布控制配置
    TOP3_HEAT_LIMIT_RATIO = 0.60
    TOP5_HEAT_LIMIT_RATIO = 0.75
    HEAT_RECYCLE_CHECK_FREQUENCY = 2
    HEAT_RECYCLE_SUPPRESSION_TURNS = 50
    HEAT_SUPPRESSION_FACTOR = 0.7
    MIN_CLUSTER_HEAT_AFTER_RECYCLE = 100
    HEAT_RECYCLE_RATE = 0.1
    
    # 分层记忆读取配置
    LAYERED_SEARCH_CONFIG = {
        "layer_1": {
            "similarity_range": (0.75, 0.80),
            "max_results": 2,
            "heat_weight_factor": 0.3,
            "frequency_weight_factor": 0.5,
            "recency_weight_factor": 0.8,
            "base_score_factor": 1.0,
            "min_heat_required": 10
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
    LAYERED_SEARCH_ENABLED = True
    LAYERED_SEARCH_FALLBACK = False
    LAYERED_SEARCH_MAX_TOTAL_RESULTS = 8
    LAYERED_SEARCH_DEDUPLICATE = True
    
    # Annoy索引配置
    ANNOY_N_TREES = 20
    ANNOY_SEARCH_K = -1
    ANNOY_REBUILD_THRESHOLD = 10
    ANNOY_METRIC = 'angular'
    
    # 冷区配置
    COLD_ZONE_MAX_MEMORIES = 100000
    COLD_ZONE_ANN_ENABLED = True
    
    # 历史记录配置
    HISTORY_FILE_PATH = "./memory/history.txt"
    HISTORY_MAX_MEMORY_RECORDS = 10000
    HISTORY_MAX_DISK_RECORDS = 1000000

# =============== 枚举和常量 ===============
class OperationType(Enum):
    MEMORY_HEAT_UPDATE = "memory_heat_update"
    CLUSTER_HEAT_UPDATE = "cluster_heat_update"
    MEMORY_TO_COLD = "memory_to_cold"
    MEMORY_TO_HOT = "memory_to_hot"
    CLUSTER_CREATE = "cluster_create"
    CLUSTER_UPDATE = "cluster_update"
    CLUSTER_DELETE = "cluster_delete"

class ConsistencyLevel(Enum):
    EVENTUAL = "eventual"
    IMMEDIATE = "immediate"
    STRONG = "strong"

# =============== 事务上下文 ===============
class TransactionContext:
    def __init__(self, memory_module, consistency_level: ConsistencyLevel = ConsistencyLevel.STRONG):
        self.memory_module = memory_module
        self.consistency_level = consistency_level
        self.operations = []
        self.memory_updates = {}
        self.cluster_updates = {}
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
    
    def add_memory_heat_update(self, memory_id: str, old_heat: int, new_heat: int, cluster_id: Optional[str] = None):
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
    id: str
    vector: np.ndarray  # 用户输入部分的向量
    user_input: str  # 用户输入部分
    ai_response: str  # AI回答部分
    summary: str = ""  # 简短摘要
    heat: int = 0
    created_turn: int = 0
    last_interaction_turn: int = 0
    access_count: int = 1
    is_hot: bool = True
    is_sleeping: bool = False
    cluster_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    update_count: int = 0
    
    @property
    def content(self) -> str:
        """向后兼容：返回完整内容"""
        return f"用户: {self.user_input}\nAI: {self.ai_response}"
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['vector'] = self.vector.tolist() if hasattr(self.vector, 'tolist') else self.vector
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        data = data.copy()
        if 'vector' in data and isinstance(data['vector'], list):
            data['vector'] = np.array(data['vector'], dtype=np.float32)
        return cls(**data)

@dataclass
class SemanticCluster:
    id: str
    centroid: np.ndarray  # 只基于用户输入向量更新
    total_heat: int = 0
    hot_memory_count: int = 0
    cold_memory_count: int = 0
    memory_ids: Set[str] = field(default_factory=set)
    is_loaded: bool = False
    size: int = 0
    last_updated_turn: int = 0
    version: int = 1
    lock: threading.RLock = field(default_factory=threading.RLock)
    pending_heat_delta: int = 0
    pending_centroid_updates: List[Tuple[np.ndarray, bool]] = field(default_factory=list)
    memory_additions_since_last_update: int = 0

# =============== 加权记忆搜索结果 ===============
@dataclass
class WeightedMemoryResult:
    memory: MemoryItem
    base_similarity: float
    relative_heat_weight: float
    access_frequency_weight: float
    recency_weight: float
    final_score: float
    ranking_position: int

# =============== 分层搜索结果 ===============
@dataclass
class LayeredSearchResult:
    layer_name: str
    similarity_range: Tuple[float, float]
    results: List[WeightedMemoryResult]
    achieved_count: int
    target_count: int
    avg_similarity: float
    avg_final_score: float

# =============== 历史记录管理器 ===============
class HistoryManager:
    """历史记录管理器，用于维护 created_turn 到记忆的映射"""
    
    def __init__(self, memory_module, history_file: str = "./memory/history.txt", 
                 max_memory_records: int = 10000, max_disk_records: int = 1000000):
        self.memory_module = memory_module
        self.history_file = Path(history_file)
        self.max_memory_records = max_memory_records  # 内存中最大记录数
        self.max_disk_records = max_disk_records      # 磁盘中最大记录数（按行数）
        
        # 内存中的映射：created_turn -> memory_id
        self.turn_to_memory_id: Dict[int, str] = {}
        
        # 最近使用的映射：memory_id -> created_turn（用于快速反向查找）
        self.memory_id_to_turn: Dict[str, int] = {}
        
        # LRU 缓存：保持最近访问的记录
        self.lru_cache: deque = deque(maxlen=max_memory_records)
        self.lru_lock = threading.RLock()
        
        # 分块管理：每1000个turn一个块
        self.block_size = 1000
        self.loaded_blocks: Set[int] = set()
        
        # 内存压缩阈值
        self.compression_threshold = max_memory_records * 2
        self.last_compression_turn = 0
        
        # 初始化历史文件
        self._init_history_file()
        
        # 加载初始数据
        self._load_recent_history()
        
        print(f"[History Manager] Initialized with {len(self.turn_to_memory_id)} records in memory")
    
    def _init_history_file(self):
        """初始化历史文件"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.history_file.exists():
            self.history_file.write_text("")
            print(f"[History Manager] Created history file: {self.history_file}")
        
        # 创建索引文件
        index_file = self.history_file.with_suffix('.idx')
        if not index_file.exists():
            index_file.write_text("")
    
    def _load_recent_history(self):
        """加载最近的历史记录"""
        try:
            if not self.history_file.exists():
                return
            
            print(f"[History Manager] Loading recent history...")
            
            # 获取文件最后N行（最近添加的记录）
            with open(self.history_file, 'r', encoding='utf-8') as f:
                lines = deque(f, maxlen=self.max_memory_records * 2)
            
            loaded_count = 0
            for line in lines:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                
                try:
                    parts = line.split(',', 2)
                    if len(parts) < 2:
                        continue
                    
                    created_turn = int(parts[0])
                    memory_id = parts[1]
                    
                    self.turn_to_memory_id[created_turn] = memory_id
                    self.memory_id_to_turn[memory_id] = created_turn
                    loaded_count += 1
                    
                    # 如果超过内存限制，停止加载
                    if len(self.turn_to_memory_id) >= self.max_memory_records:
                        print(f"[History Manager] Reached memory limit ({self.max_memory_records}), stopping load")
                        break
                        
                except (ValueError, IndexError) as e:
                    continue
            
            print(f"[History Manager] Loaded {loaded_count} recent history records")
            
        except Exception as e:
            print(f"[History Manager] Error loading history: {e}")
    
    def add_history_record(self, created_turn: int, memory_id: str, content_preview: str = ""):
        """添加历史记录"""
        # 更新内存映射
        self.turn_to_memory_id[created_turn] = memory_id
        self.memory_id_to_turn[memory_id] = created_turn
        
        # 更新LRU缓存
        with self.lru_lock:
            # 移除旧的相同记录（如果存在）
            if (created_turn, memory_id) in self.lru_cache:
                self.lru_cache.remove((created_turn, memory_id))
            self.lru_cache.append((created_turn, memory_id))
        
        # 写入文件
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                # 格式：created_turn,memory_id,content_preview
                content_preview_clean = content_preview.replace('\n', ' ').replace('\r', '')[:200]
                f.write(f"{created_turn},{memory_id},{content_preview_clean}\n")
        except Exception as e:
            print(f"[History Manager] Error writing to history file: {e}")
        
        # 检查是否需要压缩
        if len(self.turn_to_memory_id) >= self.compression_threshold:
            self._compress_memory_storage()
    
    def get_memory_by_turn(self, created_turn: int) -> Optional[MemoryItem]:
        """根据 created_turn 获取记忆"""
        # 首先检查内存映射
        memory_id = self.turn_to_memory_id.get(created_turn)
        if memory_id:
            # 更新LRU
            with self.lru_lock:
                if (created_turn, memory_id) in self.lru_cache:
                    self.lru_cache.remove((created_turn, memory_id))
                self.lru_cache.append((created_turn, memory_id))
            
            # 从内存模块获取记忆
            return self._get_memory_from_module(memory_id)
        
        # 如果不在内存中，尝试从文件加载
        memory_id = self._load_memory_id_from_file(created_turn)
        if memory_id:
            # 添加到内存映射
            self.turn_to_memory_id[created_turn] = memory_id
            self.memory_id_to_turn[memory_id] = created_turn
            
            # 更新LRU
            with self.lru_lock:
                self.lru_cache.append((created_turn, memory_id))
            
            return self._get_memory_from_module(memory_id)
        
        return None
    
    def get_turn_by_memory_id(self, memory_id: str) -> Optional[int]:
        """根据 memory_id 获取 created_turn"""
        # 检查内存映射
        if memory_id in self.memory_id_to_turn:
            created_turn = self.memory_id_to_turn[memory_id]
            
            # 更新LRU
            with self.lru_lock:
                if (created_turn, memory_id) in self.lru_cache:
                    self.lru_cache.remove((created_turn, memory_id))
                self.lru_cache.append((created_turn, memory_id))
            
            return created_turn
        
        # 如果不在内存中，可能需要从文件或数据库查询
        # 这里先返回None，调用方可以自行从数据库查询
        return None
    
    def get_memories_by_turn_range(self, start_turn: int, end_turn: int) -> List[MemoryItem]:
        """获取指定 turn 范围内的记忆"""
        memories = []
        
        # 首先检查内存中的记录
        memory_ids = []
        for turn in range(start_turn, end_turn + 1):
            if turn in self.turn_to_memory_id:
                memory_ids.append(self.turn_to_memory_id[turn])
        
        # 批量从内存模块获取
        for memory_id in memory_ids:
            memory = self._get_memory_from_module(memory_id)
            if memory:
                memories.append(memory)
        
        # 如果还需要更多记录，可以尝试从文件加载
        if len(memories) < (end_turn - start_turn + 1):
            # 可以优化：按块加载
            additional_memories = self._load_memories_from_file_range(start_turn, end_turn)
            memories.extend(additional_memories)
        
        # 按 created_turn 排序
        memories.sort(key=lambda m: m.created_turn)
        return memories
    
    def _get_memory_from_module(self, memory_id: str) -> Optional[MemoryItem]:
        """从内存模块获取记忆"""
        # 首先检查热区记忆
        memory = self.memory_module.hot_memories.get(memory_id)
        if memory:
            return memory
        
        # 检查休眠记忆
        memory = self.memory_module.sleeping_memories.get(memory_id)
        if memory:
            return memory
        
        # 从数据库查询
        try:
            cursor = self.memory_module.cursor
            cursor.execute(
                f"SELECT * FROM {self.memory_module.config.MEMORY_TABLE} WHERE id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return MemoryItem.from_dict({
                    'id': row['id'],
                    'vector': blob_to_vector(row['vector']),
                    'user_input': row['user_input'],
                    'ai_response': row['ai_response'],
                    'summary': row['summary'] or "",
                    'heat': row['heat'],
                    'created_turn': row['created_turn'],
                    'last_interaction_turn': row['last_interaction_turn'],
                    'access_count': row['access_count'],
                    'is_hot': bool(row['is_hot']),
                    'is_sleeping': bool(row['is_sleeping']),
                    'cluster_id': row['cluster_id'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'version': row['version'],
                    'update_count': row['update_count'] or 0
                })
        except Exception as e:
            print(f"[History Manager] Error querying memory from DB: {e}")
        
        return None
    
    def _load_memory_id_from_file(self, created_turn: int) -> Optional[str]:
        """从历史文件加载 memory_id"""
        try:
            block_start = (created_turn // self.block_size) * self.block_size
            block_end = block_start + self.block_size - 1
            
            # 如果这个块已经加载过，直接返回None
            if block_start in self.loaded_blocks:
                return None
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ',' not in line:
                        continue
                    
                    parts = line.split(',', 2)
                    if len(parts) < 2:
                        continue
                    
                    try:
                        turn = int(parts[0])
                        memory_id = parts[1]
                        
                        # 如果在这个块内，添加到内存映射
                        if block_start <= turn <= block_end:
                            self.turn_to_memory_id[turn] = memory_id
                            self.memory_id_to_turn[memory_id] = turn
                            
                            # 如果找到目标记录，返回
                            if turn == created_turn:
                                return memory_id
                                
                    except (ValueError, IndexError):
                        continue
            
            # 标记这个块已加载
            self.loaded_blocks.add(block_start)
            
        except Exception as e:
            print(f"[History Manager] Error loading from file: {e}")
        
        return None
    
    def _load_memories_from_file_range(self, start_turn: int, end_turn: int) -> List[MemoryItem]:
        """从文件加载指定范围的记忆"""
        memories = []
        memory_ids_found = []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ',' not in line:
                        continue
                    
                    parts = line.split(',', 2)
                    if len(parts) < 2:
                        continue
                    
                    try:
                        turn = int(parts[0])
                        memory_id = parts[1]
                        
                        if start_turn <= turn <= end_turn:
                            memory_ids_found.append(memory_id)
                            
                            # 同时更新内存映射
                            self.turn_to_memory_id[turn] = memory_id
                            self.memory_id_to_turn[memory_id] = turn
                            
                            # 如果达到内存限制，停止加载
                            if len(self.turn_to_memory_id) >= self.max_memory_records:
                                break
                                
                    except (ValueError, IndexError):
                        continue
            
            # 批量获取记忆
            for memory_id in memory_ids_found:
                memory = self._get_memory_from_module(memory_id)
                if memory:
                    memories.append(memory)
            
        except Exception as e:
            print(f"[History Manager] Error loading range from file: {e}")
        
        return memories
    
    def _compress_memory_storage(self):
        """压缩内存存储，移除不常用的记录"""
        current_turn = self.memory_module.current_turn
        
        # 每100轮才压缩一次
        if current_turn - self.last_compression_turn < 100:
            return
        
        print(f"[History Manager] Compressing memory storage...")
        
        with self.lru_lock:
            # 保留最近访问的记录
            keep_records = set(self.lru_cache)
            
            # 创建新的映射
            new_turn_to_memory_id = {}
            new_memory_id_to_turn = {}
            
            for created_turn, memory_id in keep_records:
                new_turn_to_memory_id[created_turn] = memory_id
                new_memory_id_to_turn[memory_id] = created_turn
            
            # 替换旧映射
            self.turn_to_memory_id = new_turn_to_memory_id
            self.memory_id_to_turn = new_memory_id_to_turn
            
            # 重置LRU缓存
            self.lru_cache = deque(self.lru_cache, maxlen=self.max_memory_records)
        
        self.last_compression_turn = current_turn
        
        print(f"[History Manager] Compression complete. Records: {len(self.turn_to_memory_id)}")
    
    def search_by_content_keyword(self, keyword: str, max_results: int = 50) -> List[Tuple[int, str, str]]:
        """根据内容关键词搜索历史记录（使用文件扫描）"""
        results = []
        keyword_lower = keyword.lower()
        
        try:
            # 从文件末尾开始扫描（最近的记录在前面）
            with open(self.history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 逆序处理，从最新到最旧
            for line in reversed(lines):
                line = line.strip()
                if not line or ',' not in line:
                    continue
                
                parts = line.rsplit(',', 2)
                if len(parts) < 3:
                    continue
                
                try:
                    created_turn = int(parts[0])
                    memory_id = parts[1]
                    content_preview = parts[2]
                    
                    if keyword_lower in content_preview.lower():
                        results.append((created_turn, memory_id, content_preview))
                        
                        if len(results) >= max_results:
                            break
                            
                except (ValueError, IndexError):
                    continue
        
        except Exception as e:
            print(f"[History Manager] Error searching by keyword: {e}")
        
        return results
    
    def get_history_stats(self) -> Dict[str, Any]:
        """获取历史记录统计信息"""
        try:
            # 获取文件行数
            file_lines = 0
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    file_lines = sum(1 for _ in f)
            
            # 获取时间范围
            min_turn = min(self.turn_to_memory_id.keys()) if self.turn_to_memory_id else 0
            max_turn = max(self.turn_to_memory_id.keys()) if self.turn_to_memory_id else 0
            
            return {
                'memory_records': len(self.turn_to_memory_id),
                'file_records': file_lines,
                'turn_range': (min_turn, max_turn),
                'loaded_blocks': len(self.loaded_blocks),
                'lru_cache_size': len(self.lru_cache),
                'compression_threshold': self.compression_threshold,
                'last_compression_turn': self.last_compression_turn,
                'max_memory_records': self.max_memory_records,
                'max_disk_records': self.max_disk_records
            }
            
        except Exception as e:
            print(f"[History Manager] Error getting stats: {e}")
            return {}
    
    def cleanup_old_records(self, max_age_turns: int = 100000):
        """清理过期的历史记录（基于轮数）"""
        current_turn = self.memory_module.current_turn
        cutoff_turn = current_turn - max_age_turns
        
        print(f"[History Manager] Cleaning up records older than turn {cutoff_turn} (current: {current_turn})")
        
        # 清理内存中的旧记录
        turns_to_remove = []
        for turn in self.turn_to_memory_id.keys():
            if turn < cutoff_turn:
                turns_to_remove.append(turn)
        
        for turn in turns_to_remove:
            memory_id = self.turn_to_memory_id.pop(turn, None)
            if memory_id and memory_id in self.memory_id_to_turn:
                del self.memory_id_to_turn[memory_id]
        
        # 清理LRU缓存中的旧记录
        with self.lru_lock:
            self.lru_cache = deque(
                [(t, mid) for t, mid in self.lru_cache if t >= cutoff_turn],
                maxlen=self.max_memory_records
            )
        
        print(f"[History Manager] Removed {len(turns_to_remove)} old records from memory")
    
    def rebuild_index(self):
        """重建历史索引（如果文件损坏或需要优化）"""
        print(f"[History Manager] Rebuilding history index...")
        
        # 清空当前映射
        self.turn_to_memory_id.clear()
        self.memory_id_to_turn.clear()
        self.loaded_blocks.clear()
        
        with self.lru_lock:
            self.lru_cache.clear()
        
        # 重新从文件加载
        self._load_recent_history()
        
        print(f"[History Manager] Index rebuilt. Records: {len(self.turn_to_memory_id)}")

# =============== 簇内搜索缓存 ===============
class ClusterSearchCache:
    def __init__(self, max_size: int = 50, ttl_turns: int = 100):
        self.max_size = max_size
        self.ttl_turns = ttl_turns
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_turns: Dict[str, int] = {}
        self.lock = threading.RLock()
    
    def get(self, cluster_id: str, query_vector: np.ndarray, current_turn: int) -> Optional[List[WeightedMemoryResult]]:
        with self.lock:
            if cluster_id not in self.cache:
                return None
            
            cache_entry = self.cache[cluster_id]
            
            if current_turn - cache_entry['created_turn'] > self.ttl_turns:
                del self.cache[cluster_id]
                del self.access_turns[cluster_id]
                return None
            
            cached_vector = cache_entry['query_vector']
            if not self._vectors_similar(query_vector, cached_vector):
                return None
            
            self.access_turns[cluster_id] = current_turn
            return cache_entry['results']
    
    def put(self, cluster_id: str, query_vector: np.ndarray, results: List[WeightedMemoryResult], current_turn: int):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_cluster = None
                oldest_turn = float('inf')
                for cid, aturn in self.access_turns.items():
                    if aturn < oldest_turn:
                        oldest_turn = aturn
                        oldest_cluster = cid
                
                if oldest_cluster:
                    del self.cache[oldest_cluster]
                    del self.access_turns[oldest_cluster]
            
            self.cache[cluster_id] = {
                'query_vector': query_vector.copy(),
                'results': results,
                'created_turn': current_turn
            }
            self.access_turns[cluster_id] = current_turn
    
    def clear(self, cluster_id: str = None):
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
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return False
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return similarity >= threshold

# =============== 向量缓存系统 ===============
@dataclass
class VectorCache:
    vectors: np.ndarray = None
    memory_ids: List[str] = None
    last_updated: float = 0
    is_valid: bool = False

class SimilarityCache:
    def __init__(self, max_size=100, ttl_seconds=300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, query_vector: np.ndarray) -> Optional[np.ndarray]:
        query_hash = self._hash_vector(query_vector)
        
        with self.lock:
            if query_hash not in self.cache:
                self.miss_count += 1
                return None
            
            cache_entry = self.cache[query_hash]
            
            if time.time() - cache_entry['timestamp'] > self.ttl_seconds:
                del self.cache[query_hash]
                self.miss_count += 1
                return None
            
            self.hit_count += 1
            return cache_entry['similarities']
    
    def put(self, query_vector: np.ndarray, similarities: np.ndarray):
        query_hash = self._hash_vector(query_vector)
        
        with self.lock:
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
        return hashlib.md5(vector[:8].tobytes()).hexdigest()[:16]
    
    def get_stats(self):
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
    def __init__(self):
        self.locks: Dict[str, threading.RLock] = {}
        self.lock_timestamps: Dict[str, float] = {}
        self.lock_threads: Dict[str, int] = {}
        self.global_lock = threading.RLock()
    
    def acquire(self, lock_key: str, timeout: float = 5.0) -> bool:
        start_time = time.time()
        thread_id = threading.get_ident()
        
        with self.global_lock:
            if lock_key in self.lock_threads and self.lock_threads[lock_key] == thread_id:
                return True
            
            while time.time() - start_time < timeout:
                if lock_key not in self.locks:
                    self.locks[lock_key] = threading.RLock()
                    self.lock_timestamps[lock_key] = time.time()
                    self.lock_threads[lock_key] = thread_id
                    return True
                elif self.lock_timestamps[lock_key] + timeout < time.time():
                    del self.locks[lock_key]
                    del self.lock_timestamps[lock_key]
                    if lock_key in self.lock_threads:
                        del self.lock_threads[lock_key]
                    continue
                
                time.sleep(0.01)
        
        return False
    
    def release(self, lock_key: str):
        with self.global_lock:
            thread_id = threading.get_ident()
            if lock_key in self.lock_threads and self.lock_threads[lock_key] == thread_id:
                if lock_key in self.locks:
                    del self.locks[lock_key]
                    del self.lock_timestamps[lock_key]
                del self.lock_threads[lock_key]
    
    def with_lock(self, lock_key: str, timeout: float = 5.0):
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

# =============== Annoy簇质心索引系统 ===============
class ClusterCentroidIndex:
    def __init__(self, embedding_dim: int, metric: str = 'angular', 
                 n_trees: int = 10, rebuild_threshold: int = 50):
        if not ANNOY_AVAILABLE:
            raise ImportError("Annoy is not installed. Please install: pip install annoy")
        
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.n_trees = n_trees
        self.rebuild_threshold = rebuild_threshold
        
        self.annoy_index = None
        self.index_built = False
        
        self.index_to_cluster_id: List[str] = []
        self.cluster_id_to_index: Dict[str, int] = {}
        
        self.centroid_vectors: Dict[str, np.ndarray] = {}
        
        self.changes_since_last_build = 0
        self.last_build_turn = 0
        
        self.lock = threading.RLock()
        
        self.stats = {
            'build_count': 0,
            'query_count': 0,
            'avg_query_time': 0.0,
            'total_query_time': 0.0,
            'rebuilds_due_to_changes': 0,
            'fallback_searches': 0,
            'hits': 0,
            'misses': 0
        }
        
        self._create_new_index()
    
    def _create_new_index(self):
        self.annoy_index = AnnoyIndex(self.embedding_dim, self.metric)
        self.index_built = False
        self.index_to_cluster_id.clear()
        self.cluster_id_to_index.clear()
        self.centroid_vectors.clear()
    
    def add_cluster(self, cluster_id: str, centroid: np.ndarray, current_turn: int):
        with self.lock:
            if cluster_id in self.cluster_id_to_index:
                idx = self.cluster_id_to_index[cluster_id]
                self.annoy_index.add_item(idx, centroid.tolist())
                self.centroid_vectors[cluster_id] = centroid.copy()
                self.changes_since_last_build += 1
            else:
                idx = len(self.index_to_cluster_id)
                self.index_to_cluster_id.append(cluster_id)
                self.cluster_id_to_index[cluster_id] = idx
                self.annoy_index.add_item(idx, centroid.tolist())
                self.centroid_vectors[cluster_id] = centroid.copy()
                self.changes_since_last_build += 1
            
            if self.changes_since_last_build >= self.rebuild_threshold:
                self.stats['rebuilds_due_to_changes'] += 1
    
    def remove_cluster(self, cluster_id: str):
        with self.lock:
            if cluster_id in self.cluster_id_to_index:
                self.changes_since_last_build = self.rebuild_threshold
                
                idx = self.cluster_id_to_index[cluster_id]
                self.index_to_cluster_id[idx] = None
                del self.cluster_id_to_index[cluster_id]
                
                if cluster_id in self.centroid_vectors:
                    del self.centroid_vectors[cluster_id]
                
                return True
            return False
    
    def build_index(self, force: bool = False):
        with self.lock:
            if (not force and self.changes_since_last_build == 0 and 
                self.index_built and len(self.centroid_vectors) > 0):
                return
            
            if len(self.centroid_vectors) == 0:
                self._create_new_index()
                return
            
            print(f"[Annoy Index] Building index for {len(self.centroid_vectors)} clusters...")
            start_time = time.time()
            
            self._create_new_index()
            
            self.index_to_cluster_id.clear()
            self.cluster_id_to_index.clear()
            
            idx = 0
            for cluster_id, centroid in self.centroid_vectors.items():
                self.index_to_cluster_id.append(cluster_id)
                self.cluster_id_to_index[cluster_id] = idx
                self.annoy_index.add_item(idx, centroid.tolist())
                idx += 1
            
            self.annoy_index.build(self.n_trees)
            self.index_built = True
            self.changes_since_last_build = 0
            self.last_build_turn = time.time()
            
            build_time = time.time() - start_time
            self.stats['build_count'] += 1
            
            print(f"[Annoy Index] Index built in {build_time:.2f}s with {self.n_trees} trees")
    
    def find_nearest_clusters(self, vector: np.ndarray, n: int = 5, 
                             search_k: int = -1) -> List[Tuple[str, float]]:
        start_time = time.time()
        
        with self.lock:
            if not self.index_built or len(self.centroid_vectors) == 0:
                self.stats['fallback_searches'] += 1
                return []
            
            try:
                if isinstance(vector, np.ndarray):
                    vector_list = vector.tolist()
                else:
                    vector_list = vector
                
                if search_k == -1:
                    search_k = self.n_trees * n
                
                indices, distances = self.annoy_index.get_nns_by_vector(
                    vector_list, min(n, len(self.centroid_vectors)), 
                    search_k=search_k, include_distances=True
                )
                
                results = []
                for idx, distance in zip(indices, distances):
                    if idx < len(self.index_to_cluster_id):
                        cluster_id = self.index_to_cluster_id[idx]
                        if cluster_id is None:
                            continue
                        
                        if self.metric == 'angular':
                            similarity = 1.0 - (distance ** 2) / 2.0
                        else:
                            similarity = 1.0 / (1.0 + distance)
                        
                        results.append((cluster_id, similarity))
                
                query_time = time.time() - start_time
                self.stats['query_count'] += 1
                self.stats['total_query_time'] += query_time
                self.stats['avg_query_time'] = (
                    self.stats['total_query_time'] / self.stats['query_count']
                )
                
                if results:
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
                
                return results
                
            except Exception as e:
                print(f"[Annoy Index] Query failed: {e}")
                self.stats['fallback_searches'] += 1
                self.stats['misses'] += 1
                return []
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_queries = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_queries if total_queries > 0 else 0
            
            stats = self.stats.copy()
            stats.update({
                'num_clusters': len(self.centroid_vectors),
                'index_built': self.index_built,
                'changes_since_last_build': self.changes_since_last_build,
                'embedding_dim': self.embedding_dim,
                'metric': self.metric,
                'n_trees': self.n_trees,
                'annoy_available': ANNOY_AVAILABLE,
                'hit_rate': hit_rate
            })
            return stats
    
    def clear(self):
        with self.lock:
            self._create_new_index()
            self.index_built = False
            self.changes_since_last_build = 0

# =============== 核心内存管理模块 ===============
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
        
        self._init_database()
        
        self.lock_manager = DistributedLockManager()
        
        self.hot_memories: Dict[str, MemoryItem] = {}
        self.sleeping_memories: Dict[str, MemoryItem] = {}
        self.clusters: Dict[str, SemanticCluster] = {}
        self.cluster_vectors: Dict[str, np.ndarray] = {}
        
        self.memory_to_cluster: Dict[str, str] = {}
        
        # 热力系统
        self.heat_pool: int = 0
        self.unallocated_heat: int = 0  # 未分配热力
        self.total_allocated_heat: int = 0
        self.heat_pool_lock = threading.RLock()
        
        self.clusters_needing_centroid_update: Set[str] = set()
        self.duplicate_skipped_count = 0
        
        self.update_queue = queue.Queue()
        self.operation_log: deque = deque(maxlen=10000)
        
        self.background_executor = ThreadPoolExecutor(max_workers=self.config.BACKGROUND_THREADS)
        
        self.running = True
        
        self.cluster_search_cache = ClusterSearchCache(
            max_size=self.config.CLUSTER_SEARCH_CACHE_SIZE,
            ttl_turns=self.config.CLUSTER_SEARCH_CACHE_TTL_TURNS
        )
        
        self.access_frequency_stats: Dict[str, Dict[str, Any]] = {}
        self.frequency_stats_lock = threading.RLock()
        
        self.last_heat_recycle_turn: int = 0
        self.heat_recycle_count: int = 0
        self.cluster_heat_history: Dict[str, List[Tuple[int, int]]] = {}
        
        self.vector_cache = VectorCache()
        self.vector_cache_lock = threading.RLock()
        self.similarity_cache = SimilarityCache(max_size=100, ttl_seconds=300)
        self.weight_cache: Dict[str, Dict] = {}
        self.weight_cache_turn = 0
        
        self._normalized_vectors: Optional[np.ndarray] = None
        self._precomputed_memory_norms: Optional[np.ndarray] = None
        self._precomputed_query_norms: Dict[str, float] = {}
        
        self.cluster_index = None
        if ANNOY_AVAILABLE:
            try:
                self.cluster_index = ClusterCentroidIndex(
                    embedding_dim=self.embedding_dim,
                    metric=self.config.ANNOY_METRIC,
                    n_trees=self.config.ANNOY_N_TREES,
                    rebuild_threshold=self.config.ANNOY_REBUILD_THRESHOLD
                )
                print(f"Annoy cluster centroid index initialized (dim={self.embedding_dim}, n_trees={self.config.ANNOY_N_TREES})")
            except Exception as e:
                print(f"Failed to initialize Annoy index: {e}")
                self.cluster_index = None
        else:
            print("Annoy not available. Using fallback clustering.")
        
        # 历史记录管理器
        self.history_manager = HistoryManager(
            memory_module=self,
            history_file=self.config.HISTORY_FILE_PATH,
            max_memory_records=self.config.HISTORY_MAX_MEMORY_RECORDS,
            max_disk_records=self.config.HISTORY_MAX_DISK_RECORDS
        )
        
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
            'history_file_records': 0
        }
        
        self._load_system_state()
        
        if self.cluster_index and len(self.clusters) > 0:
            self._rebuild_cluster_index()
        
        print(f"Memory Module initialized")
        print(f"NumPy vectorization and caching ENABLED")
        print(f"Annoy cluster indexing: {'ENABLED' if self.cluster_index else 'DISABLED'}")
        print(f"No background maintenance threads - all maintenance triggered by events")
        print(f"Current turn: {self.current_turn}")
        print(f"Embedding model: {'External' if self._external_embedding_func else 'Internal'}")
        print(f"Duplicate detection: {'Enabled' if self.config.DUPLICATE_CHECK_ENABLED else 'Disabled'}")
        print(f"Cluster search enabled with cache size {self.config.CLUSTER_SEARCH_CACHE_SIZE}")
        print(f"Heat distribution control enabled: Top 3 ≤ {self.config.TOP3_HEAT_LIMIT_RATIO:.0%}, "
              f"Top 5 ≤ {self.config.TOP5_HEAT_LIMIT_RATIO:.0%}")
        print(f"Layered search enabled: {'Yes' if self.config.LAYERED_SEARCH_ENABLED else 'No'}")
        print(f"Vector cache: Initialized (size: {len(self.hot_memories)})")
        print(f"Heat system: Pool={self.heat_pool:,}, Unallocated={self.unallocated_heat:,}, "
              f"Total={self.config.TOTAL_HEAT:,}")
        print(f"History Manager: Initialized with {len(self.history_manager.turn_to_memory_id)} records")
    
    # =============== 统一工具方法 ===============
    
    def _unified_update_heat(self, memory_id: str, new_heat: int, 
                            old_heat: int = None, cluster_id: str = None,
                            update_memory: bool = True, update_cluster: bool = True,
                            update_pool: bool = False, pool_delta: int = 0,
                            adjust_unallocated: bool = True) -> bool:
        """
        统一热力更新方法 - 修复：正确处理未分配热力，确保新记忆的热力被计入
        """
        memory = self.hot_memories.get(memory_id)
        if not memory:
            memory = self.sleeping_memories.get(memory_id)
        
        if not memory and old_heat is None:
            old_heat = 0
        
        if old_heat is None and memory:
            old_heat = memory.heat
        
        heat_delta = new_heat - old_heat
        
        # 重要：判断是否为新增的记忆（在hot_memories和sleeping_memories中都找不到）
        is_new_memory = (memory_id not in self.hot_memories and 
                         memory_id not in self.sleeping_memories)
        
        # 使用事务确保原子性
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            if update_memory:
                if memory:
                    memory.heat = new_heat
                    memory.update_count += 1
                tx.add_memory_heat_update(memory_id, old_heat, new_heat, cluster_id)
                
                # 更新数据库中的记忆热力
                update_memory_heat_in_db(self.cursor, self.config.MEMORY_TABLE, 
                                        memory_id, new_heat)
            
            # 只有当cluster_id有效且heat_delta不为0时才更新簇热力
            if update_cluster and cluster_id and heat_delta != 0:
                if cluster_id in self.clusters:
                    cluster = self.clusters[cluster_id]
                    with cluster.lock:
                        cluster.total_heat += heat_delta
                        # 确保热力不会变为负数
                        if cluster.total_heat < 0:
                            cluster.total_heat = 0
                    tx.add_cluster_heat_update(cluster_id, heat_delta)
                
                # 更新数据库中的簇热力
                update_cluster_heat_in_db(self.cursor, self.config.CLUSTER_TABLE, 
                                         cluster_id, heat_delta)
            
            if update_pool:
                with self.heat_pool_lock:
                    # 确保热力池不会变为负数
                    pool_change = pool_delta
                    if self.heat_pool + pool_change < 0:
                        pool_change = -self.heat_pool
                    
                    # 热力池变化时，未分配热力相应调整
                    self.heat_pool += pool_change
                    if self.heat_pool < 0:
                        self.heat_pool = 0
                    
                    # 计算总记忆热力
                    total_memory_heat = 0
                    for mem in self.hot_memories.values():
                        total_memory_heat += mem.heat
                    for mem in self.sleeping_memories.values():
                        total_memory_heat += mem.heat
                    
                    # 如果是新记忆，需要加上它的热力
                    if is_new_memory:
                        total_memory_heat += new_heat
                    
                    # 重新计算未分配热力
                    self.unallocated_heat = max(0, self.config.TOTAL_HEAT - 
                                               total_memory_heat - self.heat_pool)
                    
                    # 更新数据库中的热力池和未分配热力
                    self.cursor.execute(f"""
                        UPDATE {self.config.HEAT_POOL_TABLE}
                        SET heat_pool = ?, unallocated_heat = ?
                        WHERE id = 1
                    """, (self.heat_pool, self.unallocated_heat))
            elif adjust_unallocated:
                # 即使不更新热力池，记忆热力变化也会影响未分配热力
                with self.heat_pool_lock:
                    # 计算总记忆热力
                    total_memory_heat = 0
                    for mem in self.hot_memories.values():
                        total_memory_heat += mem.heat
                    for mem in self.sleeping_memories.values():
                        total_memory_heat += mem.heat
                    
                    # 如果是新记忆，需要加上它的热力
                    if is_new_memory:
                        total_memory_heat += new_heat
                    
                    # 重新计算未分配热力
                    self.unallocated_heat = max(0, self.config.TOTAL_HEAT - 
                                               total_memory_heat - self.heat_pool)
                    
                    self.cursor.execute(f"""
                        UPDATE {self.config.HEAT_POOL_TABLE}
                        SET unallocated_heat = ?
                        WHERE id = 1
                    """, (self.unallocated_heat,))
        
        # 使相关缓存失效
        self._invalidate_related_caches(memory_id, cluster_id)
        
        self.operation_count += 1
        self._trigger_maintenance_if_needed()
        
        return True
    
    
    def _unified_centroid_management(self, cluster_id: str, vector: np.ndarray,
                                   operation: str, memory_id: str = None):
        """
        统一簇质心管理
        """
        if cluster_id in self.clusters:
            schedule_centroid_update(
                self.clusters, cluster_id, vector, 
                self.clusters_needing_centroid_update,
                add=(operation == 'add')
            )
        
        if self.cluster_index and ANNOY_AVAILABLE:
            if operation == 'add':
                self.cluster_index.add_cluster(cluster_id, vector, self.current_turn)
            elif operation == 'remove':
                self.cluster_index.remove_cluster(cluster_id)
        
        if operation == 'add':
            self.memory_additions_since_last_centroid_update += 1
        
        self._invalidate_related_caches(memory_id, cluster_id)
    
    def _invalidate_related_caches(self, memory_id: str = None, 
                                 cluster_id: str = None, full: bool = False):
        """
        统一失效相关缓存
        """
        if full:
            self.weight_cache.clear()
            self.cluster_search_cache.clear()
            self.similarity_cache.cache.clear()
            self.invalidate_vector_cache()
            return
        
        if memory_id:
            if memory_id in self.weight_cache:
                del self.weight_cache[memory_id]
            
            query_keys_to_remove = []
            for query_hash in self.similarity_cache.cache.keys():
                if memory_id in query_hash:
                    query_keys_to_remove.append(query_hash)
            
            for key in query_keys_to_remove:
                if key in self.similarity_cache.cache:
                    del self.similarity_cache.cache[key]
            
            self.invalidate_vector_cache(memory_id)
        
        if cluster_id:
            self.cluster_search_cache.clear(cluster_id)
            
            keys_to_remove = []
            for mem_id, weights in self.weight_cache.items():
                if mem_id in self.hot_memories:
                    mem = self.hot_memories[mem_id]
                    if mem.cluster_id == cluster_id:
                        keys_to_remove.append(mem_id)
            
            for key in keys_to_remove:
                del self.weight_cache[key]
    
    def _unified_turn_increment(self, increment_type: str = 'access') -> int:
        """
        统一轮数递增
        """
        increment = {
            'access': self.config.TURN_INCREMENT_ON_ACCESS,
            'add': self.config.TURN_INCREMENT_ON_ADD,
            'maintenance': self.config.TURN_INCREMENT_ON_MAINTENANCE
        }.get(increment_type, 1)
        
        self.current_turn = increment_turn(
            self.current_turn, self.stats, self.turn_lock, increment
        )
        
        return self.current_turn
    
    def _unified_vector_storage(self, vector: np.ndarray, 
                               operation: str = 'store') -> bytes:
        """
        统一向量存储/加载
        """
        if operation == 'store':
            return vector_to_blob(vector)
        else:
            return blob_to_vector(vector) if isinstance(vector, bytes) else vector
    
    def _update_unallocated_heat(self):
        """更新未分配热力值"""
        with self.heat_pool_lock:
            total_allocated = (self.heat_pool + 
                              sum(m.heat for m in self.hot_memories.values()) +
                              sum(m.heat for m in self.sleeping_memories.values()))
            
            self.unallocated_heat = max(0, self.config.TOTAL_HEAT - total_allocated)
            
            # 更新数据库
            self.cursor.execute(f"""
                UPDATE {self.config.HEAT_POOL_TABLE}
                SET unallocated_heat = ?
                WHERE id = 1
            """, (self.unallocated_heat,))
        
        return self.unallocated_heat
    
    def find_best_clusters_for_query(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        根据查询文本找到最相关的语义簇
        """
        # 获取查询的向量表示
        query_vector = self._get_embedding(query)
        
        # 使用Annoy索引或线性搜索找到最相关的簇
        if self.cluster_index and ANNOY_AVAILABLE:
            try:
                results = self.cluster_index.find_nearest_clusters(
                    query_vector, 
                    n=min(top_k, len(self.clusters)),
                    search_k=self.config.ANNOY_SEARCH_K
                )
                
                # 确保簇存在于当前系统中
                valid_results = []
                for cluster_id, similarity in results:
                    if cluster_id in self.clusters:
                        valid_results.append((cluster_id, similarity))
                
                if valid_results:
                    return valid_results[:top_k]
            except Exception as e:
                print(f"[Cluster Search] Annoy search failed: {e}")
                # 降级到线性搜索
        
        # 线性搜索：计算查询向量与每个簇质心的相似度
        results = []
        for cluster_id, cluster in self.clusters.items():
            if cluster.centroid is not None:
                similarity = compute_cosine_similarity(query_vector, cluster.centroid)
                results.append((cluster_id, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def search_in_clusters(self, query_text: str, top_clusters: int = 3, results_per_cluster: int = 4) -> List[WeightedMemoryResult]:
        """
        在多个簇中搜索记忆
        """
        query_vector = self._get_embedding(query_text)
        
        # 找到最相关的簇
        best_clusters = self.find_best_clusters_for_query(query_text, top_k=top_clusters)
        
        all_results = []
        
        for cluster_id, cluster_similarity in best_clusters:
            # 在每个簇内搜索
            cluster_results = self.search_within_cluster(
                query_vector=query_vector,
                cluster_id=cluster_id,
                max_results=results_per_cluster * 2  # 获取更多，然后筛选
            )
            
            # 添加簇级别的相似度作为基础分数的一部分
            for result in cluster_results:
                # 结合簇相似度和记忆相似度
                combined_similarity = 0.7 * result.base_similarity + 0.3 * cluster_similarity
                result.base_similarity = combined_similarity
                result.final_score = combined_similarity * np.exp(np.mean(np.log([
                    max(0.0001, result.relative_heat_weight),
                    max(0.0001, result.access_frequency_weight),
                    max(0.0001, result.recency_weight)
                ])))
        
            all_results.extend(cluster_results)
        
        # 按最终分数排序
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # 返回前 N 个结果
        max_results = top_clusters * results_per_cluster
        return all_results[:max_results]
    
    # =============== 重构后的核心方法 ===============
    
    def add_memory(self, user_input: str, ai_response: str, metadata: Dict[str, Any] = None) -> str:
        """添加记忆 - 新版本：分别存储用户输入和AI回答"""
        return self._add_memory_internal(user_input=user_input, ai_response=ai_response, metadata=metadata)
    
    def _add_memory_internal(self, user_input: str, ai_response: str, metadata: Dict[str, Any] = None) -> str:
        """内部方法：添加记忆"""
        current_turn = self._unified_turn_increment('add')
        
        # 只使用用户输入部分计算向量
        user_vector = self._get_embedding(user_input)
        
        # 生成摘要
        summary = self._generate_summary(user_input, ai_response)
        
        # 检查重复（基于用户输入）
        duplicate_id = self._check_duplicate(user_vector, user_input)
        if duplicate_id:
            if duplicate_id in self.hot_memories:
                memory = self.hot_memories[duplicate_id]
                memory.last_interaction_turn = current_turn
                memory.access_count += 1
                
                self._unified_update_heat(
                    duplicate_id, memory.heat,
                    update_memory=False, update_cluster=False, update_pool=False
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
        
        # 检查热力池是否需要回收（低于阈值时）
        pool_threshold = self.config.INITIAL_HEAT_POOL * self.config.HEAT_POOL_RECYCLE_THRESHOLD
        if self.heat_pool < pool_threshold:
            print(f"[Heat Pool] Low pool ({self.heat_pool:,} < {pool_threshold:,}), triggering recycle")
            self._recycle_heat_pool()
        
        # 检查热力池是否足够
        if self.heat_pool < self.config.NEW_MEMORY_HEAT:
            # 如果热力池不足，尝试回收更多
            need_heat = self.config.NEW_MEMORY_HEAT - self.heat_pool
            self._recycle_from_memories(need_heat)
        
        memory = MemoryItem(
            id=memory_id,
            vector=user_vector,  # 只使用用户输入向量
            user_input=user_input,
            ai_response=ai_response,
            summary=summary,
            heat=0,
            created_turn=current_turn,
            last_interaction_turn=current_turn,
            metadata=metadata or {}
        )
        
        # 确定要分配的总热力A
        base_allocated_heat = min(self.config.NEW_MEMORY_HEAT, self.heat_pool)
        suppression_factor = self._get_suppression_factor()
        
        if suppression_factor < 1.0:
            allocated_heat = int(base_allocated_heat * suppression_factor)
        else:
            allocated_heat = base_allocated_heat
        
        # 总热力A = 新记忆A/2 + 邻居总A/2
        # 先为新记忆预分配A热力（从热力池取出）
        self._unified_update_heat(
            memory_id=memory_id,
            new_heat=allocated_heat,  # 先给新记忆分配A热力
            old_heat=0,
            update_pool=True,
            pool_delta=-allocated_heat  # 从热力池取出A
        )
        
        neighbors = self._find_neighbors(user_vector, exclude_id=memory_id)
        
        if neighbors:
            neighbor_count = min(len(neighbors), 5)  # 最多5个邻居
            # 邻居总热力 = A/2
            total_neighbor_heat = allocated_heat // 2
            # 每个邻居得到的热力 = (A/2) / neighbor_count
            heat_per_neighbor = total_neighbor_heat // neighbor_count if neighbor_count > 0 else 0
            # 新记忆最终得到的热力 = A - 给邻居的总热力
            new_memory_final_heat = allocated_heat - total_neighbor_heat
            
            # 调整新记忆的热力：从A减少到A/2
            self._unified_update_heat(
                memory_id=memory_id,
                new_heat=new_memory_final_heat,
                old_heat=allocated_heat,
                update_pool=False,  # 不更新热力池，只是内部转移
                adjust_unallocated=True
            )
            
            # 给邻居分配热力
            for (neighbor_id, _, neighbor_memory) in neighbors[:neighbor_count]:
                new_neighbor_heat = neighbor_memory.heat + heat_per_neighbor
                neighbor_cluster_id = neighbor_memory.cluster_id
                
                self._unified_update_heat(
                    memory_id=neighbor_id,
                    new_heat=new_neighbor_heat,
                    old_heat=neighbor_memory.heat,
                    cluster_id=neighbor_cluster_id,
                    update_memory=True,
                    update_cluster=(neighbor_cluster_id is not None),
                    update_pool=False,
                    adjust_unallocated=True
                )
            
            memory.heat = new_memory_final_heat
        else:
            # 没有邻居，新记忆得到全部A热力
            memory.heat = allocated_heat
        
        # 分配到簇（使用用户输入向量）
        cluster_id = self._assign_to_cluster(memory, user_vector)
        
        # 更新簇质心（使用用户输入向量）
        self._unified_centroid_management(
            cluster_id=cluster_id,
            vector=user_vector,
            operation='add',
            memory_id=memory_id
        )
        
        self.hot_memories[memory_id] = memory
        self.memory_to_cluster[memory_id] = cluster_id
        
        self._update_access_frequency(memory_id)
        
        # 将新记忆的热力添加到簇中
        if cluster_id and cluster_id in self.clusters:
            cluster = self.clusters[cluster_id]
            with cluster.lock:
                cluster.total_heat += memory.heat
                cluster.hot_memory_count += 1
                cluster.size += 1
                cluster.memory_ids.add(memory_id)

            # 更新数据库中的簇热力
            update_cluster_heat_in_db(self.cursor, self.config.CLUSTER_TABLE, cluster_id, memory.heat)
        
        # 插入到数据库
        self.cursor.execute(f"""
            INSERT INTO {self.config.MEMORY_TABLE} 
            (id, vector, user_input, ai_response, summary, heat, created_turn, last_interaction_turn, 
             access_count, is_hot, is_sleeping, cluster_id, metadata, update_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
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
            json.dumps(memory.metadata)
        ))
        
        # 添加到历史记录管理器
        self.history_manager.add_history_record(
            created_turn=current_turn,
            memory_id=memory_id,
            content_preview=summary
        )
        
        self.stats['hot_memories'] += 1
        self.stats['total_memories'] += 1
        self.memory_addition_count += 1
        self._trigger_maintenance_if_needed()
        
        with self.heat_pool_lock:
            current_pool = self.heat_pool
            current_unallocated = self.unallocated_heat
        
        print(f"[Memory] Added memory {memory_id} with {memory.heat:,} heat "
              f"(Pool: {current_pool:,}, Unallocated: {current_unallocated:,}, "
              f"Pool Used: {allocated_heat:,})")
        
        return memory_id
    
    def _generate_summary(self, user_input: str, ai_response: str) -> str:
        """生成简短摘要"""
        # 简单的摘要生成逻辑
        user_preview = user_input[:50] + ("..." if len(user_input) > 50 else "")
        ai_preview = ai_response[:50] + ("..." if len(ai_response) > 50 else "")
        return f"用户: {user_preview} | AI: {ai_preview}"
    
    def access_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """重构后的访问记忆方法"""
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
    
    def _recycle_heat_pool(self):
        """重构后的热力回收方法 - 修复：保持热力池稳定"""
        self._unified_turn_increment()
        
        target_pool_size = self.config.INITIAL_HEAT_POOL
        current_need = target_pool_size - self.heat_pool
        
        if current_need <= 0:
            return
        
        print(f"[Heat Pool] Need to recycle {current_need:,} heat to refill pool "
              f"(current: {self.heat_pool:,}, target: {target_pool_size:,})")
        
        # 首先从未分配热力中提取
        if self.unallocated_heat >= current_need:
            with self.heat_pool_lock:
                transfer = min(self.unallocated_heat, current_need)
                self.heat_pool += transfer
                self.unallocated_heat -= transfer
                
                self.cursor.execute(f"""
                    UPDATE {self.config.HEAT_POOL_TABLE}
                    SET heat_pool = ?, unallocated_heat = ?
                    WHERE id = 1
                """, (self.heat_pool, self.unallocated_heat))
            
            print(f"[Heat Pool] Transferred {transfer:,} heat from unallocated to pool")
            self.stats['heat_recycled_to_pool'] += transfer
            return
        
        # 如果未分配热力不足，从记忆中回收
        if self.unallocated_heat > 0:
            with self.heat_pool_lock:
                transfer = self.unallocated_heat
                self.heat_pool += transfer
                self.unallocated_heat = 0
                current_need -= transfer
                
                self.cursor.execute(f"""
                    UPDATE {self.config.HEAT_POOL_TABLE}
                    SET heat_pool = ?, unallocated_heat = ?
                    WHERE id = 1
                """, (self.heat_pool, self.unallocated_heat))
            
            print(f"[Heat Pool] Transferred {transfer:,} heat from unallocated to pool")
            self.stats['heat_recycled_to_pool'] += transfer
        
        # 从记忆中回收剩余需要的热力
        if current_need > 0:
            self._recycle_from_memories(current_need)
    
    def _recycle_from_memories(self, need_heat: int):
        """从记忆中回收指定数量的热力 - 修复：正确处理未分配热力"""
        if need_heat <= 0:
            return
        
        print(f"[Heat Pool] Recycling {need_heat:,} heat from memories")
        
        eligible_memories = []
        for memory_id, memory in self.hot_memories.items():
            if memory.heat > 10 and not memory.is_sleeping:  # 保留最小热力
                eligible_memories.append((memory_id, memory))
        
        if not eligible_memories:
            print(f"[Heat Pool] No eligible memories for recycling")
            return
        
        # 按热力降序排序，从高热度记忆中回收
        eligible_memories.sort(key=lambda x: x[1].heat, reverse=True)
        
        recycled = 0
        for memory_id, memory in eligible_memories:
            if recycled >= need_heat:
                break
            
            # 计算可从该记忆中回收的热力（保留最小热力）
            max_recyclable = max(0, memory.heat - 10)
            to_recycle = min(max_recyclable, need_heat - recycled)
            
            if to_recycle <= 0:
                continue
            
            new_heat = memory.heat - to_recycle
            
            # 使用统一方法更新记忆热力，热力回收回到热力池
            self._unified_update_heat(
                memory_id=memory_id,
                new_heat=new_heat,
                old_heat=memory.heat,
                cluster_id=memory.cluster_id,
                update_memory=True,
                update_cluster=True,
                update_pool=True,
                pool_delta=to_recycle  # 热力池增加
            )
            
            recycled += to_recycle
            print(f"[Heat Pool] Recycled {to_recycle:,} heat from memory {memory_id[:8]}... "
                  f"(new heat: {new_heat:,})")
        
        print(f"[Heat Pool] Total recycled: {recycled:,} heat")
        self.stats['total_heat_recycled'] += recycled
        
        # 更新未分配热力
        self._update_unallocated_heat()
    
    def _check_and_move_sleeping(self):
        """重构后的休眠记忆移动方法 - 修复：热力正确回收"""
        if len(self.sleeping_memories) == 0:
            return
        
        print(f"Moving {len(self.sleeping_memories)} sleeping memories to cold zone")
        
        for memory_id, memory in list(self.sleeping_memories.items()):
            # 先将热力回收回热力池
            if memory.heat > 0:
                self._unified_update_heat(
                    memory_id=memory_id,
                    new_heat=0,
                    old_heat=memory.heat,
                    cluster_id=memory.cluster_id,
                    update_memory=True,
                    update_cluster=True,
                    update_pool=True,
                    pool_delta=memory.heat  # 热力池增加
                )
            
            memory.is_hot = False
            memory.is_sleeping = False
            
            if memory.cluster_id:
                self._unified_centroid_management(
                    cluster_id=memory.cluster_id,
                    vector=memory.vector,
                    operation='remove',
                    memory_id=memory_id
                )
                
                # 更新簇统计
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
    
    # =============== 原始方法（使用新工具函数重构） ===============
    
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
            warnings.warn(f"sentence-transformers not installed. Using random embeddings.")
            self.model = None
            self.embedding_dim = self.config.EMBEDDING_DIM
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}. Using random embeddings.")
            self.model = None
            self.embedding_dim = self.config.EMBEDDING_DIM
    
    def _init_database(self):
        """初始化数据库表结构 - 修改：添加user_input, ai_response, summary字段"""
        self.conn = sqlite3.connect(self.config.DB_PATH, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # 修改表结构：拆分content字段为user_input, ai_response, summary
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.MEMORY_TABLE} (
                id TEXT PRIMARY KEY,
                vector BLOB,
                user_input TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                summary TEXT,
                heat INTEGER DEFAULT 0,
                created_turn INTEGER DEFAULT 0,
                last_interaction_turn INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 1,
                is_hot INTEGER DEFAULT 1,
                is_sleeping INTEGER DEFAULT 0,
                cluster_id TEXT,
                metadata TEXT,
                version INTEGER DEFAULT 1,
                update_count INTEGER DEFAULT 0
            )
        """)
        
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
        
        # 修复：添加未分配热力字段
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.HEAT_POOL_TABLE} (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                heat_pool INTEGER DEFAULT {self.config.INITIAL_HEAT_POOL},
                unallocated_heat INTEGER DEFAULT {self.config.TOTAL_HEAT - self.config.INITIAL_HEAT_POOL},  -- 新增
                total_allocated_heat INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                current_turn INTEGER DEFAULT {self.config.INITIAL_TURN}
            )
        """)
        
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.OPERATION_LOG_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                operation_type TEXT,
                memory_id TEXT,
                cluster_id TEXT,
                old_value TEXT,
                new_value TEXT,
                turn INTEGER DEFAULT 0,
                applied INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0
            )
        """)
        
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_op_log_turn ON {self.config.OPERATION_LOG_TABLE}(turn)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_op_log_applied ON {self.config.OPERATION_LOG_TABLE}(applied)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_cluster ON {self.config.MEMORY_TABLE}(cluster_id, heat)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_hot_heat ON {self.config.MEMORY_TABLE}(is_hot, heat DESC)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_turn ON {self.config.MEMORY_TABLE}(last_interaction_turn DESC)")
        
        # 修复：插入语句包含未分配热力
        self.cursor.execute(f"""
            INSERT OR IGNORE INTO {self.config.HEAT_POOL_TABLE} 
            (id, heat_pool, unallocated_heat, total_allocated_heat, current_turn)
            VALUES (1, {self.config.INITIAL_HEAT_POOL}, 
                   {self.config.TOTAL_HEAT - self.config.INITIAL_HEAT_POOL}, 0, {self.config.INITIAL_TURN})
        """)
        
        self.conn.commit()
    
    def _load_system_state(self):
        """加载系统状态 - 修复：加载未分配热力"""
        self.cursor.execute(
            f"SELECT heat_pool, unallocated_heat, total_allocated_heat, current_turn "
            f"FROM {self.config.HEAT_POOL_TABLE} WHERE id = 1"
        )
        row = self.cursor.fetchone()
        if row:
            self.heat_pool = row['heat_pool']
            self.unallocated_heat = row['unallocated_heat']  # 新增
            self.total_allocated_heat = row['total_allocated_heat']
            self.current_turn = row['current_turn']
            self.stats['current_turn'] = self.current_turn
        
        self._load_all_clusters()
        self._load_hot_memories()
        self._load_access_frequency_stats()
        self._check_consistency()
        
        print(f"System state loaded. Current turn: {self.current_turn}")
        print(f"Heat system: Pool={self.heat_pool:,}, Unallocated={self.unallocated_heat:,}, "
              f"Memory heat={sum(m.heat for m in self.hot_memories.values()):,}")
    
    def _load_all_clusters(self):
        """加载所有语义簇"""
        self.cursor.execute(f"SELECT * FROM {self.config.CLUSTER_TABLE}")
        rows = self.cursor.fetchall()
        
        for row in rows:
            cluster = SemanticCluster(
                id=row['id'],
                centroid=blob_to_vector(row['centroid']),
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
                update_count=row['update_count'] or 0
            )
            
            self.hot_memories[memory.id] = memory
            self.memory_to_cluster[memory.id] = memory.cluster_id
            
            if memory.cluster_id and memory.cluster_id in self.clusters:
                cluster = self.clusters[memory.cluster_id]
                cluster.memory_ids.add(memory.id)
                if not cluster.is_loaded:
                    cluster.is_loaded = True
        
        print(f"Loaded {len(self.hot_memories)} hot memories from database")
        self._rebuild_vector_cache()
    
    def _load_access_frequency_stats(self):
        """加载访问频率统计"""
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
        """检查系统一致性 - 修复：验证热力守恒"""
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
        
        # 验证热力守恒
        expected_total_allocated = total_heat_in_memories + self.heat_pool + self.unallocated_heat
        expected_total_heat = self.config.TOTAL_HEAT
        
        if abs(expected_total_allocated - expected_total_heat) > 100:  # 允许微小误差
            print(f"WARNING: Heat conservation violated! "
                  f"Total allocated: {expected_total_allocated:,}, "
                  f"Expected total: {expected_total_heat:,}")
            self.stats['consistency_violations'] += 1
            self._repair_consistency()
        
        # 验证记忆热力与簇热力匹配
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
              f"Total={total_heat_in_memories + self.heat_pool + self.unallocated_heat:,}")
        print(f"Current turn: {self.current_turn}")
    
    def _repair_consistency(self):
        """修复一致性 - 修复：考虑未分配热力"""
        print("Attempting to repair consistency...")
        
        # 重新计算每个簇的热力
        for cluster_id, cluster in self.clusters.items():
            # 从数据库中重新加载该簇的所有记忆
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
                
                # 更新簇对象
                cluster.total_heat = db_total_heat
                cluster.hot_memory_count = db_hot_count
                cluster.cold_memory_count = db_cold_count
                cluster.size = db_total_count
                
                # 更新数据库中的簇信息
                self.cursor.execute(f"""
                    UPDATE {self.config.CLUSTER_TABLE}
                    SET total_heat = ?, hot_memory_count = ?, 
                        cold_memory_count = ?, size = ?, version = version + 1
                    WHERE id = ?
                """, (db_total_heat, db_hot_count, db_cold_count, db_total_count, cluster_id))
                
                print(f"  Cluster {cluster_id[:8]}...: heat={db_total_heat:,}, "
                      f"hot={db_hot_count}, cold={db_cold_count}")
        
        # 重新计算总记忆热力
        self.cursor.execute(f"SELECT SUM(heat) as total_heat FROM {self.config.MEMORY_TABLE}")
        row = self.cursor.fetchone()
        total_memory_heat = row['total_heat'] or 0
        
        # 重新计算未分配热力
        allocated_heat = total_memory_heat + self.heat_pool
        self.unallocated_heat = max(0, self.config.TOTAL_HEAT - allocated_heat)
        
        # 确保热力池不超过初始大小（除非未分配热力不足）
        if self.heat_pool > self.config.INITIAL_HEAT_POOL and self.unallocated_heat < 0:
            excess = self.heat_pool - self.config.INITIAL_HEAT_POOL
            self.heat_pool -= excess
            self.unallocated_heat += excess
        
        # 更新数据库
        with self.heat_pool_lock:
            self.cursor.execute(f"""
                UPDATE {self.config.HEAT_POOL_TABLE}
                SET heat_pool = ?, unallocated_heat = ?
                WHERE id = 1
            """, (self.heat_pool, self.unallocated_heat))
        
        print(f"  Heat pool: {self.heat_pool:,}")
        print(f"  Unallocated heat: {self.unallocated_heat:,}")
        print(f"  Total memory heat: {total_memory_heat:,}")
        print("Consistency repair completed")
    
    def _audit_heat_balance(self) -> Dict[str, Any]:
        """审计热力平衡 - 检测泄漏"""
        total_hot_heat = sum(m.heat for m in self.hot_memories.values())
        total_sleeping_heat = sum(m.heat for m in self.sleeping_memories.values())
        total_cold_heat = 0  # 冷区记忆热力为0
        
        total_in_system = (self.heat_pool + 
                          self.unallocated_heat + 
                          total_hot_heat + 
                          total_sleeping_heat + 
                          total_cold_heat)
        
        expected_total = self.config.TOTAL_HEAT
        discrepancy = total_in_system - expected_total
        
        audit_result = {
            'heat_pool': self.heat_pool,
            'unallocated_heat': self.unallocated_heat,
            'hot_memories_heat': total_hot_heat,
            'sleeping_memories_heat': total_sleeping_heat,
            'cold_memories_heat': total_cold_heat,
            'total_in_system': total_in_system,
            'expected_total': expected_total,
            'discrepancy': discrepancy,
            'has_leak': abs(discrepancy) > 100,  # 允许100的误差
            'hot_memory_count': len(self.hot_memories),
            'sleeping_memory_count': len(self.sleeping_memories),
            'current_turn': self.current_turn
        }
        
        if abs(discrepancy) > 100:
            print(f"[HEAT AUDIT] LEAK DETECTED! Discrepancy: {discrepancy:,}")
            print(f"  Expected: {expected_total:,}")
            print(f"  Actual: {total_in_system:,}")
            print(f"  Breakdown: Pool={self.heat_pool:,}, Unallocated={self.unallocated_heat:,}, "
                  f"Hot={total_hot_heat:,}, Sleeping={total_sleeping_heat:,}")
            
            # 自动修复：调整未分配热力
            if discrepancy > 0:
                # 系统中有多余热力，减少未分配热力
                self.unallocated_heat = max(0, self.unallocated_heat - discrepancy)
            else:
                # 系统中缺少热力，增加未分配热力
                self.unallocated_heat += abs(discrepancy)
            
            # 更新数据库
            self.cursor.execute(f"""
                UPDATE {self.config.HEAT_POOL_TABLE}
                SET unallocated_heat = ?
                WHERE id = 1
            """, (self.unallocated_heat,))
            
            print(f"[HEAT AUDIT] Auto-repaired. New unallocated heat: {self.unallocated_heat:,}")
        
        return audit_result
    
    def _ensure_transaction(self, transaction_id: str):
        """确保事务开始"""
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
        """完成事务"""
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
        need_maintenance = False
        
        if self.operation_count >= self.MAINTENANCE_OPERATION_THRESHOLD:
            need_maintenance = True
            self.stats['events_triggered'] += 1
        
        if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
            need_maintenance = True
            self.stats['events_triggered'] += 1
        
        if need_maintenance:
            self._unified_turn_increment('maintenance')
            self.background_executor.submit(self._perform_maintenance_tasks)
            
            if self.operation_count >= self.MAINTENANCE_OPERATION_THRESHOLD:
                self.operation_count = 0
            if self.memory_addition_count >= self.CHECKPOINT_MEMORY_THRESHOLD:
                self.memory_addition_count = 0
    
    def _perform_maintenance_tasks(self):
        """执行维护任务 - 添加热力审计"""
        current_turn = self.current_turn
        print(f"\n[Memory System] Performing maintenance tasks (Turn: {current_turn})")
        
        start_time = time.time()
        
        # 1. 热力审计
        audit_result = self._audit_heat_balance()
        if audit_result['has_leak']:
            print(f"[Maintenance] Heat leak detected and repaired")
        
        # 2. 其他维护任务...
        self._flush_update_queue()
        self._apply_pending_cluster_updates()
        
        if self.memory_additions_since_last_centroid_update >= self.config.CENTROID_UPDATE_FREQUENCY:
            self._update_cluster_centroids_batch()
        
        self._check_consistency()
        self._check_and_adjust_heat_distribution()
        self._create_checkpoint_if_needed()
        
        if len(self.sleeping_memories) > 0:
            self._check_and_move_sleeping()
        
        self._update_memory_cache_state()
        self._cleanup_access_frequency_stats()
        self._cleanup_cluster_heat_history()
        
        if self.cluster_index and self.cluster_index.changes_since_last_build >= self.config.ANNOY_REBUILD_THRESHOLD:
            self._rebuild_cluster_index()
        
        elapsed = time.time() - start_time
        self.stats['maintenance_cycles'] += 1
        
        print(f"[Memory System] Maintenance cycle {self.stats['maintenance_cycles']} completed in {elapsed:.2f}s")
    
    def _check_and_adjust_heat_distribution(self):
        """检查并调整热力分布"""
        self.maintenance_cycles_since_heat_check += 1
        if self.maintenance_cycles_since_heat_check < self.config.HEAT_RECYCLE_CHECK_FREQUENCY:
            return
        
        self.maintenance_cycles_since_heat_check = 0
        
        print(f"[Heat Distribution] Checking cluster heat distribution (Turn: {self.current_turn})")
        
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
                
                if cluster_id not in self.cluster_heat_history:
                    self.cluster_heat_history[cluster_id] = []
                self.cluster_heat_history[cluster_id].append((self.current_turn, cluster.total_heat))
        
        if total_cluster_heat == 0 or len(cluster_heat_list) <= 5:
            print(f"[Heat Distribution] Not enough clusters or total heat is zero")
            return
        
        cluster_heat_list.sort(key=lambda x: x['heat'], reverse=True)
        
        top3_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:3])
        top5_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:5])
        
        top3_ratio = top3_heat / total_cluster_heat
        top5_ratio = top5_heat / total_cluster_heat
        
        print(f"[Heat Distribution] Top 3 clusters: {top3_ratio:.2%} (limit: {self.config.TOP3_HEAT_LIMIT_RATIO:.0%})")
        print(f"[Heat Distribution] Top 5 clusters: {top5_ratio:.2%} (limit: {self.config.TOP5_HEAT_LIMIT_RATIO:.0%})")
        
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
        
        print(f"[Heat Distribution] Starting heat redistribution...")
        self._redistribute_cluster_heat(cluster_heat_list, total_cluster_heat)
        
        self.last_heat_recycle_turn = self.current_turn
        self.heat_recycle_count += 1
        self.stats['heat_redistributions'] = self.stats.get('heat_redistributions', 0) + 1
        
        print(f"[Heat Distribution] Heat redistribution completed at turn {self.current_turn}")
    
    def _redistribute_cluster_heat(self, cluster_heat_list: List[Dict], total_cluster_heat: int):
        """重新分配簇热力"""
        total_size = sum(cluster['size'] for cluster in cluster_heat_list)
        if total_size == 0:
            return
        
        excess_heat = 0
        
        top3_excess = max(0, sum(cluster['heat'] for cluster in cluster_heat_list[:3]) - 
                         total_cluster_heat * self.config.TOP3_HEAT_LIMIT_RATIO)
        
        top5_excess = max(0, sum(cluster['heat'] for cluster in cluster_heat_list[:5]) - 
                         total_cluster_heat * self.config.TOP5_HEAT_LIMIT_RATIO)
        
        excess_heat = max(top3_excess, top5_excess)
        
        if excess_heat <= 0:
            return
        
        print(f"[Heat Distribution] Excess heat to recycle: {excess_heat:,}")
        
        total_top_heat = sum(cluster['heat'] for cluster in cluster_heat_list[:5])
        if total_top_heat == 0:
            return
        
        recycled_heat = 0
        
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            for i, cluster_info in enumerate(cluster_heat_list[:5]):
                cluster_id = cluster_info['cluster_id']
                cluster = self.clusters.get(cluster_id)
                if not cluster:
                    continue
                
                cluster_excess_ratio = cluster.total_heat / total_top_heat
                cluster_excess_heat = int(excess_heat * cluster_excess_ratio * self.config.HEAT_RECYCLE_RATE)
                
                min_heat_for_cluster = max(self.config.MIN_CLUSTER_HEAT_AFTER_RECYCLE, 
                                         cluster.size * 10)
                
                if cluster.total_heat - cluster_excess_heat < min_heat_for_cluster:
                    cluster_excess_heat = max(0, cluster.total_heat - min_heat_for_cluster)
                
                if cluster_excess_heat <= 0:
                    continue
                
                print(f"[Heat Distribution] Recycling {cluster_excess_heat:,} heat from cluster {cluster_id[:8]}...")
                
                memories_to_adjust = []
                for memory_id in list(cluster.memory_ids):
                    if memory_id in self.hot_memories:
                        memories_to_adjust.append(self.hot_memories[memory_id])
                
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
                    
                    update_memory_heat_in_db(self.cursor, self.config.MEMORY_TABLE, memory.id, new_heat)
                    
                    recycled_heat += heat_to_deduct
                
                cluster.total_heat -= cluster_excess_heat
                tx.add_cluster_heat_update(cluster_id, -cluster_excess_heat)
                
                update_cluster_heat_in_db(self.cursor, self.config.CLUSTER_TABLE, cluster_id, -cluster_excess_heat)
            
            if recycled_heat > 0:
                with self.heat_pool_lock:
                    old_heat_pool = self.heat_pool
                    self.heat_pool += recycled_heat
                    
                    self.cursor.execute(f"""
                        UPDATE {self.config.HEAT_POOL_TABLE}
                        SET heat_pool = ?
                        WHERE id = 1
                    """, (self.heat_pool,))
                
                print(f"[Heat Distribution] Recycled {recycled_heat:,} heat back to pool. "
                      f"Pool: {old_heat_pool:,} -> {self.heat_pool:,}")
                
                self.stats['heat_recycled_to_pool'] = self.stats.get('heat_recycled_to_pool', 0) + recycled_heat
        
        for cluster_info in cluster_heat_list[:5]:
            cluster_id = cluster_info['cluster_id']
            self.cluster_search_cache.clear(cluster_id)
        
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
        
        turns_since_recycle = self.current_turn - self.last_heat_recycle_turn
        remaining_suppression = max(0, self.config.HEAT_RECYCLE_SUPPRESSION_TURNS - turns_since_recycle)
        
        suppression_factor = self.config.HEAT_SUPPRESSION_FACTOR + (
            (1.0 - self.config.HEAT_SUPPRESSION_FACTOR) * 
            (1.0 - remaining_suppression / self.config.HEAT_RECYCLE_SUPPRESSION_TURNS)
        )
        
        return min(1.0, max(self.config.HEAT_SUPPRESSION_FACTOR, suppression_factor))
    
    def _cleanup_access_frequency_stats(self):
        """清理过期的访问频率统计"""
        with self.frequency_stats_lock:
            memory_ids_to_remove = []
            for memory_id in self.access_frequency_stats:
                if memory_id not in self.hot_memories:
                    memory_ids_to_remove.append(memory_id)
            
            for memory_id in memory_ids_to_remove:
                del self.access_frequency_stats[memory_id]
    
    def _cleanup_cluster_heat_history(self):
        """清理过期的簇热力历史记录"""
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
        """创建检查点 - 修复：保存未分配热力"""
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
            
            with self.heat_pool_lock:
                self.cursor.execute(f"""
                    UPDATE {self.config.HEAT_POOL_TABLE}
                    SET heat_pool = ?, unallocated_heat = ?, total_allocated_heat = ?, 
                        version = version + 1, current_turn = ?
                    WHERE id = 1
                """, (self.heat_pool, self.unallocated_heat, self.total_allocated_heat, self.current_turn))
            
            self.conn.commit()
            self.memory_addition_count = 0
            
            print(f"[Memory System] Checkpoint created successfully")
            print(f"  Heat pool: {self.heat_pool:,}")
            print(f"  Unallocated heat: {self.unallocated_heat:,}")
        except Exception as e:
            print(f"[Memory System] Error creating checkpoint: {e}")
            self.conn.rollback()
    
    def _update_memory_cache_state(self):
        """更新内存缓存状态"""
        pass
    
    def _validate_transaction(self, operations: List[Dict]) -> bool:
        """验证事务的一致性约束"""
        total_heat_delta = 0
        cluster_heat_deltas = defaultdict(int)
        
        for op in operations:
            if op['type'] == OperationType.MEMORY_HEAT_UPDATE:
                heat_delta = op['new_heat'] - op['old_heat']
                total_heat_delta += heat_delta
                
                if op['cluster_id']:
                    cluster_heat_deltas[op['cluster_id']] += heat_delta
        
        return True
    
    def _apply_operation(self, operation: Dict, immediate: bool = False):
        """应用单个操作"""
        op_type = operation['type']
        
        try:
            if op_type == OperationType.MEMORY_HEAT_UPDATE:
                self._apply_memory_heat_update(operation, immediate)
            elif op_type == OperationType.CLUSTER_HEAT_UPDATE:
                self._apply_cluster_heat_update(operation, immediate)
            
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
                    self._update_cluster_heat(cluster_id, heat_delta, immediate)
        
        if memory_id in self.weight_cache:
            del self.weight_cache[memory_id]
    
    def _apply_cluster_heat_update(self, operation: Dict, immediate: bool):
        """应用簇热力更新"""
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
        """更新簇热力"""
        if immediate:
            with self.lock_manager.with_lock(f"cluster_{cluster_id}", self.config.CLUSTER_LOCK_TIMEOUT):
                if cluster_id in self.clusters:
                    self.clusters[cluster_id].total_heat += heat_delta
                
                update_cluster_heat_in_db(self.cursor, self.config.CLUSTER_TABLE, cluster_id, heat_delta)
        else:
            self.update_queue.put({
                'type': 'cluster_heat_update',
                'cluster_id': cluster_id,
                'heat_delta': heat_delta,
                'turn': self.current_turn
            })
        
        self.weight_cache.clear()
    
    def _update_memory_and_cluster_heat_atomic(self, memory_id: str, new_heat: int, 
                                              cluster_id: str = None, 
                                              old_heat: int = None) -> bool:
        """原子更新记忆和簇的热力"""
        if old_heat is None and memory_id in self.hot_memories:
            old_heat = self.hot_memories[memory_id].heat
        
        if old_heat is None:
            self.cursor.execute(f"SELECT heat FROM {self.config.MEMORY_TABLE} WHERE id = ?", (memory_id,))
            row = self.cursor.fetchone()
            old_heat = row['heat'] if row else 0
        
        heat_delta = new_heat - old_heat
        
        with TransactionContext(self, ConsistencyLevel.STRONG) as tx:
            tx.add_memory_heat_update(memory_id, old_heat, new_heat, cluster_id)
            
            if cluster_id and heat_delta != 0:
                tx.add_cluster_heat_update(cluster_id, heat_delta)
        
        return True
    
    def _process_batch_updates(self, batch: List[Dict]):
        """批量处理更新"""
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
        
        with self.conn:
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
        
        for memory_id, new_heat in memory_updates.items():
            if memory_id in self.hot_memories:
                self.hot_memories[memory_id].heat = new_heat
        
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
                    self.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET total_heat = total_heat + ?, pending_heat_delta = 0
                        WHERE id = ?
                    """, (pending_delta, cluster_id))
                    
                    if cluster_id in self.clusters:
                        with self.clusters[cluster_id].lock:
                            self.clusters[cluster_id].total_heat += pending_delta
                            self.clusters[cluster_id].pending_heat_delta = 0
            
            self.weight_cache.clear()
    
    def _check_duplicate(self, vector: np.ndarray, content: str = None) -> Optional[str]:
        """检查是否为重复记忆"""
        if not self.config.DUPLICATE_CHECK_ENABLED:
            return None
        
        best_similarity = 0.0
        best_memory_id = None
        best_memory = None
        
        for memory_id, memory in self.hot_memories.items():
            similarity = compute_cosine_similarity(vector, memory.vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_memory_id = memory_id
                best_memory = memory
            
            if best_similarity >= self.config.DUPLICATE_THRESHOLD:
                break
        
        if content and best_similarity < self.config.DUPLICATE_THRESHOLD:
            content_hash = hashlib.md5(content.strip().lower().encode()).hexdigest()
            for memory_id, memory in self.hot_memories.items():
                memory_hash = hashlib.md5(memory.user_input.strip().lower().encode()).hexdigest()
                if content_hash == memory_hash:
                    return memory_id
        
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
    
    def _update_cluster_centroids_batch(self):
        """批量更新簇质心"""
        if self.memory_additions_since_last_centroid_update < self.config.CENTROID_UPDATE_FREQUENCY:
            return
        
        print(f"[Memory System] Updating cluster centroids after {self.memory_additions_since_last_centroid_update} memory additions")
        
        self.memory_additions_since_last_centroid_update = 0
        
        clusters_to_update = list(self.clusters_needing_centroid_update)
        self.clusters_needing_centroid_update.clear()
        
        if not clusters_to_update:
            for cluster_id, cluster in self.clusters.items():
                if cluster.memory_additions_since_last_update > 0:
                    clusters_to_update.append(cluster_id)
        
        batch_size = self.config.CENTROID_UPDATE_BATCH_SIZE
        for i in range(0, len(clusters_to_update), batch_size):
            batch = clusters_to_update[i:i+batch_size]
            self._update_cluster_centroids(batch)
    
    def _update_cluster_centroids(self, cluster_ids: List[str]):
        """更新指定簇的质心 - 只基于用户输入向量"""
        centroid_updates = {}
        
        for cluster_id in cluster_ids:
            if cluster_id not in self.clusters:
                continue
            
            cluster = self.clusters[cluster_id]
            if cluster.memory_additions_since_last_update == 0 and not cluster.pending_centroid_updates:
                continue
            
            with cluster.lock:
                if cluster.memory_additions_since_last_update >= self.config.CENTROID_FULL_RECALC_THRESHOLD:
                    new_centroid = self._recalculate_cluster_centroid(cluster_id)
                    self.stats['full_centroid_recalculations'] += 1
                else:
                    new_centroid = self._incremental_update_cluster_centroid(cluster)
                
                if new_centroid is not None:
                    cluster.centroid = new_centroid
                    cluster.memory_additions_since_last_update = 0
                    cluster.pending_centroid_updates.clear()
                    cluster.last_updated_turn = self.current_turn
                    cluster.version += 1
                    
                    self.cluster_vectors[cluster_id] = new_centroid
                    
                    self._update_cluster_index(cluster_id, new_centroid, 'update')
                    
                    centroid_updates[cluster_id] = {
                        'centroid': new_centroid,
                        'turn': self.current_turn
                    }
        
        if centroid_updates:
            with self.conn:
                for cluster_id, update_data in centroid_updates.items():
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
            
            self.stats['centroid_updates'] += len(centroid_updates)
            print(f"[Memory System] Updated centroids for {len(centroid_updates)} clusters")
            
            for cluster_id in cluster_ids:
                self.cluster_search_cache.clear(cluster_id)
    
    def _incremental_update_cluster_centroid(self, cluster: SemanticCluster) -> Optional[np.ndarray]:
        """增量更新簇质心 - 只基于用户输入向量"""
        if not cluster.pending_centroid_updates and cluster.memory_additions_since_last_update == 0:
            return None
        
        if cluster.pending_centroid_updates:
            new_centroid = cluster.centroid.copy()
            
            for vector, add in cluster.pending_centroid_updates:
                if add:
                    if cluster.size > 0:
                        new_centroid = (new_centroid * cluster.size + vector) / (cluster.size + 1)
                    else:
                        new_centroid = vector
                    cluster.size += 1
                else:
                    if cluster.size > 1:
                        new_centroid = (new_centroid * cluster.size - vector) / (cluster.size - 1)
                    else:
                        new_centroid = np.zeros(self.embedding_dim, dtype=np.float32)
                    cluster.size -= 1
            
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            return new_centroid
        
        return None
    
    def _recalculate_cluster_centroid(self, cluster_id: str) -> Optional[np.ndarray]:
        """完全重算簇质心 - 只基于用户输入向量"""
        if cluster_id not in self.clusters:
            return None
        
        self.cursor.execute(f"""
            SELECT vector FROM {self.config.MEMORY_TABLE}
            WHERE cluster_id = ? AND is_hot = 1
            LIMIT 1000
        """, (cluster_id,))
        
        rows = self.cursor.fetchall()
        if not rows:
            return None
        
        vectors = [blob_to_vector(row['vector']) for row in rows]
        
        new_centroid = np.mean(vectors, axis=0)
        
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        
        return new_centroid
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入"""
        if self._external_embedding_func is not None:
            return self._external_embedding_func(text)
        elif self.model:
            try:
                return self.model.encode(text, show_progress_bar=False)
            except:
                return self.model.encode(text)
        else:
            if not hasattr(self, 'embedding_dim'):
                self.embedding_dim = self.config.EMBEDDING_DIM
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if self._external_similarity_func is not None:
            return self._external_similarity_func(vec1, vec2)
        
        return compute_cosine_similarity(vec1, vec2)
    
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
                
                if len(stats['recent_interactions']) > 100:
                    stats['recent_interactions'] = stats['recent_interactions'][-100:]
                
                if self.current_turn - stats['last_reset_turn'] > 1000:
                    stats['count'] = 1
                    stats['last_reset_turn'] = self.current_turn
                    stats['recent_interactions'] = [self.current_turn]
        
        if memory_id in self.weight_cache:
            del self.weight_cache[memory_id]
    
    def _get_access_frequency_weight(self, memory_id: str, memory_item: MemoryItem) -> float:
        """获取访问频率权重"""
        with self.frequency_stats_lock:
            if memory_id not in self.access_frequency_stats:
                return 1.0
            
            stats = self.access_frequency_stats[memory_id]
            access_count = stats['count']
            
            recent_interactions = [turn for turn in stats['recent_interactions'] 
                                  if self.current_turn - turn < 1000]
            recent_count = len(recent_interactions)
            
            total_factor = min(1.0, self.config.ACCESS_FREQUENCY_DISCOUNT_THRESHOLD / max(1, access_count))
            recent_factor = min(1.0, self.config.ACCESS_FREQUENCY_DISCOUNT_THRESHOLD / max(1, recent_count))
            
            weight = 0.3 * total_factor + 0.7 * recent_factor
            
            if memory_item.is_sleeping:
                weight *= 0.5
            
            return max(0.1, weight)
    
    def _get_recency_weight(self, memory_item: MemoryItem) -> float:
        """获取最近访问权重"""
        turns_since_interaction = self.current_turn - memory_item.last_interaction_turn
        
        weight = 1.0 - (turns_since_interaction * self.config.RECENCY_WEIGHT_DECAY_PER_TURN)
        
        return max(0.1, min(1.0, weight))
    
    def _get_relative_heat_weight(self, memory_item: MemoryItem, cluster_total_heat: int) -> float:
        """获取相对热力权重"""
        if cluster_total_heat <= 0:
            return 1.0
        
        relative_heat = memory_item.heat / cluster_total_heat
        
        weight = relative_heat ** self.config.RELATIVE_HEAT_WEIGHT_POWER
        
        weight = max(0.1, min(1.0, weight))
        
        return weight
    
    def _rebuild_vector_cache(self):
        """重建向量缓存"""
        with self.vector_cache_lock:
            memory_ids, vectors = convert_memory_vectors(list(self.hot_memories.values()))
            
            if vectors.shape[0] > 0:
                self.vector_cache.vectors = vectors
            else:
                self.vector_cache.vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
            
            self.vector_cache.memory_ids = memory_ids
            self.vector_cache.last_updated = time.time()
            self.vector_cache.is_valid = True
            
            self._normalized_vectors = None
            self._precomputed_memory_norms = None
            
            print(f"[Vector Cache] Rebuilt cache with {len(memory_ids)} vectors")
    
    def _ensure_vector_cache(self):
        """确保向量缓存是最新的"""
        with self.vector_cache_lock:
            if (self.vector_cache.is_valid and 
                self.vector_cache.vectors is not None and
                len(self.vector_cache.memory_ids) == len(self.hot_memories)):
                return
            
            self._rebuild_vector_cache()
    
    def _compute_all_similarities_vectorized(self, query_vector: np.ndarray) -> np.ndarray:
        """向量化计算所有相似度"""
        self._ensure_vector_cache()
        
        vectors = self.vector_cache.vectors
        if vectors.shape[0] == 0:
            return np.array([])
        
        return compute_batch_similarities(query_vector, vectors)
    
    def _get_cached_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """获取缓存的相似度"""
        cached_similarities = self.similarity_cache.get(query_vector)
        if cached_similarities is not None:
            self.stats['similarity_cache_hits'] += 1
            return cached_similarities
        
        self.stats['similarity_cache_misses'] += 1
        
        similarities = self._compute_all_similarities_vectorized(query_vector)
        self.similarity_cache.put(query_vector, similarities)
        
        return similarities
    
    def _ensure_weight_cache(self):
        """确保权重缓存是最新的"""
        current_turn = self.current_turn
        
        if (current_turn - self.weight_cache_turn > 100 or
            len(self.weight_cache) != len(self.hot_memories)):
            
            with self.frequency_stats_lock:
                self.weight_cache.clear()
                
                for memory_id, memory in self.hot_memories.items():
                    cluster_total_heat = self.clusters[memory.cluster_id].total_heat if memory.cluster_id in self.clusters else 1
                    
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
                if memory_id in self.vector_cache.memory_ids:
                    idx = self.vector_cache.memory_ids.index(memory_id)
                    memory = self.hot_memories.get(memory_id)
                    if memory:
                        self.vector_cache.vectors[idx] = memory.vector
                        
                        self._normalized_vectors = None
                        self._precomputed_memory_norms = None
            else:
                self.vector_cache.is_valid = False
        
        self.similarity_cache.cache.clear()
    
    # =============== Annoy索引相关方法 ===============
    
    def _find_best_cluster_annoy(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        """使用Annoy索引查找最相似的簇"""
        if self.cluster_index is None or len(self.clusters) < 3:
            return self._find_best_cluster_linear(vector)
        
        try:
            start_time = time.time()
            results = self.cluster_index.find_nearest_clusters(
                vector, 
                n=min(5, len(self.clusters)),
                search_k=self.config.ANNOY_SEARCH_K
            )
            query_time = time.time() - start_time
            
            self.stats['annoy_queries'] += 1
            
            if query_time > 0.01:
                print(f"[Annoy] Query took {query_time*1000:.1f}ms for {len(self.clusters)} clusters")
            
            if not results:
                self.stats['annoy_fallback_searches'] += 1
                return self._find_best_cluster_linear(vector)
            
            best_cluster_id, best_similarity = results[0]
            
            if best_cluster_id in self.clusters:
                cluster = self.clusters[best_cluster_id]
                if cluster.centroid is not None:
                    actual_similarity = self._compute_similarity(vector, cluster.centroid)
                    return best_cluster_id, actual_similarity
            
            self.stats['annoy_fallback_searches'] += 1
            return self._find_best_cluster_linear(vector)
                
        except Exception as e:
            print(f"[Annoy] Search failed: {e}")
            self.stats['annoy_fallback_searches'] += 1
            return self._find_best_cluster_linear(vector)
    
    def _find_best_cluster_linear(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        """线性搜索最相似的簇"""
        best_cluster_id = None
        best_similarity = -1.0
        
        for cluster_id, centroid in self.cluster_vectors.items():
            similarity = self._compute_similarity(vector, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id
        
        return best_cluster_id, best_similarity
    
    def _update_cluster_index(self, cluster_id: str, centroid: np.ndarray = None, 
                             operation: str = 'add'):
        """更新Annoy索引中的簇"""
        if self.cluster_index is None or not ANNOY_AVAILABLE:
            return
        
        if centroid is None and cluster_id in self.clusters:
            cluster = self.clusters[cluster_id]
            centroid = cluster.centroid
        
        if centroid is not None:
            if operation == 'add' or operation == 'update':
                self.cluster_index.add_cluster(cluster_id, centroid, self.current_turn)
            elif operation == 'remove':
                self.cluster_index.remove_cluster(cluster_id)
    
    def _rebuild_cluster_index(self):
        """重建簇质心索引"""
        if self.cluster_index is None:
            return
        
        if self.cluster_index.changes_since_last_build < self.config.ANNOY_REBUILD_THRESHOLD:
            return
        
        print(f"[Annoy Index] Rebuilding index for {len(self.clusters)} clusters...")
        
        self.cluster_index.clear()
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.centroid is not None:
                self.cluster_index.add_cluster(cluster_id, cluster.centroid, self.current_turn)
        
        self.cluster_index.build_index(force=True)
    
    def _assign_to_cluster(self, memory: MemoryItem, vector: np.ndarray) -> str:
        """分配到簇"""
        best_cluster_id, best_similarity = self._find_best_cluster_annoy(vector)
        
        if best_similarity >= self.config.CLUSTER_SIMILARITY_THRESHOLD:
            cluster_id = best_cluster_id
        else:
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
            
            self._update_cluster_index(cluster_id, vector, 'add')
            
            self.stats['clusters'] += 1
            
            self._save_cluster_to_db(cluster)
        
        cluster = self.clusters[cluster_id]
        with cluster.lock:
            cluster.memory_ids.add(memory.id)
            cluster.size += 1
            cluster.hot_memory_count += 1
            cluster.is_loaded = True
        
        memory.cluster_id = cluster_id
        return cluster_id
    
    def _save_cluster_to_db(self, cluster: SemanticCluster):
        """保存簇到数据库"""
        self.cursor.execute(f"""
            INSERT INTO {self.config.CLUSTER_TABLE} 
            (id, centroid, total_heat, hot_memory_count, cold_memory_count, 
             is_loaded, size, last_updated_turn, memory_additions_since_last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cluster.id,
            vector_to_blob(cluster.centroid),
            cluster.total_heat,
            cluster.hot_memory_count,
            cluster.cold_memory_count,
            int(cluster.is_loaded),
            cluster.size,
            cluster.last_updated_turn,
            cluster.memory_additions_since_last_update
        ))
    
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
            vector_to_blob(cluster.centroid),
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
    
    def _find_neighbors(self, vector: np.ndarray, exclude_id: str = None, limit: int = None) -> List[Tuple[str, float, MemoryItem]]:
        """寻找相似邻居"""
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
    
    def _exponential_allocation(self, similarities: List[float], total_heat: int) -> List[int]:
        """指数函数分配热力"""
        if not similarities:
            return []
        
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
    
    # =============== 历史记录查询方法 ===============
    
    def get_memory_by_turn(self, created_turn: int) -> Optional[MemoryItem]:
        """根据 created_turn 获取记忆"""
        return self.history_manager.get_memory_by_turn(created_turn)
    
    def get_turn_by_memory_id(self, memory_id: str) -> Optional[int]:
        """根据 memory_id 获取 created_turn"""
        return self.history_manager.get_turn_by_memory_id(memory_id)
    
    def get_memories_by_turn_range(self, start_turn: int, end_turn: int) -> List[MemoryItem]:
        """获取指定 turn 范围内的记忆"""
        return self.history_manager.get_memories_by_turn_range(start_turn, end_turn)
    
    def search_history_by_keyword(self, keyword: str, max_results: int = 50) -> List[Tuple[int, str, str]]:
        """根据关键词搜索历史记录"""
        return self.history_manager.search_by_content_keyword(keyword, max_results)
    
    def get_history_stats(self) -> Dict[str, Any]:
        """获取历史记录统计信息"""
        return self.history_manager.get_history_stats()
    
    def cleanup_old_history(self, max_age_turns: int = 100000):
        """清理过期的历史记录"""
        return self.history_manager.cleanup_old_records(max_age_turns)
    
    def rebuild_history_index(self):
        """重建历史索引"""
        return self.history_manager.rebuild_index()
    
    def search_layered_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                               max_total_results: int = None,
                               config_override: Dict = None) -> Dict[str, LayeredSearchResult]:
        """优化的分层搜索"""
        if not self.config.LAYERED_SEARCH_ENABLED:
            warnings.warn("Layered search is disabled. Using default search.")
            return self._fallback_search(query_text, query_vector, max_total_results)
        
        if query_vector is None and query_text is not None:
            query_vector = self._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")
        
        config = config_override or self.config.LAYERED_SEARCH_CONFIG
        
        if max_total_results is None:
            max_total_results = self.config.LAYERED_SEARCH_MAX_TOTAL_RESULTS
        
        self.stats['layered_searches'] += 1
        self.stats['vectorized_searches'] += 1
        
        start_time = time.time()
        similarities = self._get_cached_similarities(query_vector)
        sim_compute_time = time.time() - start_time
        
        self._ensure_weight_cache()
        
        memory_ids = self.vector_cache.memory_ids
        layered_results = {}
        
        seen_memory_ids = set() if self.config.LAYERED_SEARCH_DEDUPLICATE else None
        
        for layer_name in ["layer_3", "layer_2", "layer_1"]:
            if layer_name not in config:
                continue
            
            layer_config = config[layer_name]
            similarity_min, similarity_max = layer_config["similarity_range"]
            max_results = layer_config["max_results"]
            min_heat = layer_config.get("min_heat_required", 0)
            
            mask = (similarities >= similarity_min) & (similarities < similarity_max)
            candidate_indices = np.where(mask)[0]
            
            if candidate_indices.size == 0:
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
            
            if seen_memory_ids is not None and candidate_indices.size > 0:
                valid_indices = []
                for idx in candidate_indices:
                    memory_id = memory_ids[idx]
                    if memory_id not in seen_memory_ids:
                        valid_indices.append(idx)
                candidate_indices = np.array(valid_indices)
            
            layer_results = []
            candidates_processed = 0
            
            sorted_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]
            
            for idx in sorted_indices[:max_results]:
                memory_id = memory_ids[idx]
                memory = self.hot_memories[memory_id]
                
                if memory.heat < min_heat:
                    continue
                
                cached_weights = self.weight_cache.get(memory_id, {})
                if not cached_weights:
                    cluster_total_heat = self.clusters[memory.cluster_id].total_heat if memory.cluster_id in self.clusters else 1
                    
                    relative_heat_weight = self._get_relative_heat_weight(memory, cluster_total_heat)
                    access_frequency_weight = self._get_access_frequency_weight(memory_id, memory)
                    recency_weight = self._get_recency_weight(memory)
                else:
                    relative_heat_weight = cached_weights['relative_heat_weight']
                    access_frequency_weight = cached_weights['access_frequency_weight']
                    recency_weight = cached_weights['recency_weight']
                
                heat_weight_factor = layer_config.get("heat_weight_factor", 1.0)
                frequency_weight_factor = layer_config.get("frequency_weight_factor", 1.0)
                recency_weight_factor = layer_config.get("recency_weight_factor", 1.0)
                base_score_factor = layer_config.get("base_score_factor", 1.0)
                
                adj_relative_heat_weight = relative_heat_weight * heat_weight_factor
                adj_access_frequency_weight = access_frequency_weight * frequency_weight_factor
                adj_recency_weight = recency_weight * recency_weight_factor
                adj_base_similarity = similarities[idx] * base_score_factor
                
                weights = [adj_relative_heat_weight, adj_access_frequency_weight, adj_recency_weight]
                weights_nonzero = [max(0.0001, w) for w in weights]
                geometric_mean = np.exp(np.mean(np.log(weights_nonzero)))
                
                final_score = adj_base_similarity * geometric_mean
                
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
                
                if seen_memory_ids is not None:
                    seen_memory_ids.add(memory_id)
                
                if candidates_processed >= max_results:
                    break
            
            achieved_count = len(layer_results)
            if achieved_count > 0:
                avg_similarity = np.mean([r.base_similarity for r in layer_results])
                avg_final_score = np.mean([r.final_score for r in layer_results])
            else:
                avg_similarity = 0.0
                avg_final_score = 0.0
            
            layered_results[layer_name] = LayeredSearchResult(
                layer_name=layer_name,
                similarity_range=(similarity_min, similarity_max),
                results=layer_results,
                achieved_count=achieved_count,
                target_count=max_results,
                avg_similarity=avg_similarity,
                avg_final_score=avg_final_score
            )
            
            total_results = sum(len(r.results) for r in layered_results.values())
            if total_results >= max_total_results:
                break
        
        total_time = time.time() - start_time
        if total_time > 0.1:
            print(f"[Performance] Vectorized layered search: {total_time*1000:.1f}ms "
                  f"(similarity: {sim_compute_time*1000:.1f}ms)")
        
        return layered_results
    
    def _fallback_search(self, query_text: str = None, query_vector: np.ndarray = None,
                        max_total_results: int = None) -> Dict[str, LayeredSearchResult]:
        """后备搜索方法"""
        if query_vector is None and query_text is not None:
            query_vector = self._get_embedding(query_text)
        
        if max_total_results is None:
            max_total_results = 8
        
        all_results = self.search_similar_memories(
            query_vector=query_vector,
            max_results=max_total_results,
            use_weighting=True
        )
        
        layered_results = {}
        default_layers = {
            "layer_1": {"similarity_range": (0.0, 1.0), "results": [], "count": 0},
            "layer_2": {"similarity_range": (0.0, 1.0), "results": [], "count": 0},
            "layer_3": {"similarity_range": (0.0, 1.0), "results": [], "count": 0}
        }
        
        for result in all_results:
            if result.base_similarity >= 0.85:
                default_layers["layer_3"]["results"].append(result)
            elif result.base_similarity >= 0.80:
                default_layers["layer_2"]["results"].append(result)
            elif result.base_similarity >= 0.75:
                default_layers["layer_1"]["results"].append(result)
        
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
                target_count=0,
                avg_similarity=avg_similarity,
                avg_final_score=avg_final_score
            )
        
        return layered_results
    
    def get_layered_search_results(self, query_text: str = None, query_vector: np.ndarray = None,
                                  flatten_results: bool = True) -> List[WeightedMemoryResult]:
        """
        获取分层搜索结果（扁平化版本）
        """
        layered_results = self.search_layered_memories(
            query_text=query_text,
            query_vector=query_vector
        )
        
        if not flatten_results:
            return layered_results
        
        flattened = []
        for layer_name in ["layer_3", "layer_2", "layer_1"]:
            if layer_name in layered_results:
                layer_result = layered_results[layer_name]
                flattened.extend(layer_result.results)
        
        return flattened
    
    def update_layered_search_config(self, new_config: Dict = None, **kwargs):
        """更新分层搜索配置"""
        if new_config:
            self.config.LAYERED_SEARCH_CONFIG = new_config
        else:
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
    
    def search_within_cluster(self, query_text: str = None, query_vector: np.ndarray = None, 
                             cluster_id: str = None, max_results: int = None) -> List[WeightedMemoryResult]:
        """
        在指定簇内搜索相似记忆
        """
        if max_results is None:
            max_results = self.config.CLUSTER_SEARCH_MAX_RESULTS
        
        if query_vector is None and query_text is not None:
            query_vector = self._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")
        
        if cluster_id is None:
            if self.cluster_index and len(self.clusters) >= 3:
                results = self.cluster_index.find_nearest_clusters(
                    query_vector, 
                    n=1,
                    search_k=self.config.ANNOY_SEARCH_K
                )
                if results:
                    cluster_id = results[0][0]
            else:
                best_cluster_id, best_similarity = self._find_best_cluster_linear(query_vector)
                cluster_id = best_cluster_id
        
        if cluster_id is None or cluster_id not in self.clusters:
            return []
        
        cached_results = self.cluster_search_cache.get(cluster_id, query_vector, self.current_turn)
        if cached_results is not None:
            self.stats['cache_hits'] += 1
            return cached_results[:max_results]
        
        self.stats['cache_misses'] += 1
        self.stats['cluster_searches'] += 1
        
        cluster = self.clusters[cluster_id]
        cluster_total_heat = cluster.total_heat
        
        weighted_results = []
        
        cluster_memory_ids = set()
        
        with cluster.lock:
            cluster_memory_ids.update(cluster.memory_ids)
        
        for memory_id, memory in self.hot_memories.items():
            if memory.cluster_id == cluster_id:
                cluster_memory_ids.add(memory_id)
        
        for memory_id in cluster_memory_ids:
            memory = self.hot_memories.get(memory_id)
            if memory is None or memory.is_sleeping:
                continue
            
            base_similarity = compute_cosine_similarity(query_vector, memory.vector)
            
            if base_similarity < self.config.SIMILARITY_THRESHOLD:
                continue
            
            relative_heat_weight = self._get_relative_heat_weight(memory, cluster_total_heat)
            access_frequency_weight = self._get_access_frequency_weight(memory_id, memory)
            recency_weight = self._get_recency_weight(memory)
            
            weights = [relative_heat_weight, access_frequency_weight, recency_weight]
            geometric_mean = np.exp(np.mean(np.log([max(0.0001, w) for w in weights])))
            
            final_score = base_similarity * geometric_mean
            
            result = WeightedMemoryResult(
                memory=memory,
                base_similarity=base_similarity,
                relative_heat_weight=relative_heat_weight,
                access_frequency_weight=access_frequency_weight,
                recency_weight=recency_weight,
                final_score=final_score,
                ranking_position=0
            )
            
            weighted_results.append(result)
        
        weighted_results.sort(key=lambda x: x.final_score, reverse=True)
        
        for i, result in enumerate(weighted_results[:max_results]):
            result.ranking_position = i + 1
        
        final_results = weighted_results[:max_results]
        
        self.stats['weight_adjustments'] += len(final_results)
        
        high_freq_count = sum(1 for r in final_results 
                             if r.access_frequency_weight < 0.5)
        self.stats['high_frequency_memories'] += high_freq_count
        
        if final_results:
            self.cluster_search_cache.put(cluster_id, query_vector, final_results, self.current_turn)
        
        return final_results
    
    def search_similar_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                               max_results: int = 10, use_weighting: bool = True) -> List[WeightedMemoryResult]:
        """
        搜索相似记忆（跨所有簇）
        """
        if query_vector is None and query_text is not None:
            query_vector = self._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")
        
        all_results = []
        
        if use_weighting:
            for cluster_id in self.clusters.keys():
                cluster_results = self.search_within_cluster(
                    query_vector=query_vector,
                    cluster_id=cluster_id,
                    max_results=max_results // 2
                )
                all_results.extend(cluster_results)
        else:
            for memory_id, memory in self.hot_memories.items():
                if memory.is_sleeping:
                    continue
                
                similarity = compute_cosine_similarity(query_vector, memory.vector)
                
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
        
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        
        for i, result in enumerate(all_results[:max_results]):
            result.ranking_position = i + 1
        
        return all_results[:max_results]
    
    def get_cluster_statistics(self, cluster_id: str) -> Dict[str, Any]:
        """获取簇的统计信息"""
        if cluster_id not in self.clusters:
            return {}
        
        cluster = self.clusters[cluster_id]
        memories_in_cluster = []
        
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
        
        heat_values = [m.heat for m in memories_in_cluster]
        total_heat = sum(heat_values)
        
        heat_distribution = []
        for memory in memories_in_cluster:
            if total_heat > 0:
                relative_heat = memory.heat / total_heat
            else:
                relative_heat = 0.0
            
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
        
        heat_distribution.sort(key=lambda x: x['heat'], reverse=True)
        
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
            'heat_distribution': heat_distribution[:10],
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
                for mid in list(self.access_frequency_stats.keys()):
                    self.access_frequency_stats[mid] = {
                        'count': 1,
                        'last_reset_turn': self.current_turn,
                        'recent_interactions': [self.current_turn]
                    }
        
        self.weight_cache.clear()
    
    def adjust_memory_weights(self, memory_id: str, 
                             heat_adjustment: float = 1.0,
                             frequency_adjustment: float = 1.0):
        """手动调整记忆权重"""
        if memory_id not in self.hot_memories:
            return False
        
        memory = self.hot_memories[memory_id]
        
        if heat_adjustment != 1.0:
            new_heat = int(memory.heat * heat_adjustment)
            self._unified_update_heat(
                memory_id=memory_id,
                new_heat=new_heat,
                old_heat=memory.heat
            )
        
        if frequency_adjustment != 1.0:
            with self.frequency_stats_lock:
                if memory_id in self.access_frequency_stats:
                    stats = self.access_frequency_stats[memory_id]
                    adjusted_count = int(stats['count'] * frequency_adjustment)
                    stats['count'] = max(1, adjusted_count)
        
        if memory.cluster_id:
            self.cluster_search_cache.clear(memory.cluster_id)
        
        if memory_id in self.weight_cache:
            del self.weight_cache[memory_id]
        
        return True
    
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
        
        annoy_stats = {}
        if self.cluster_index:
            annoy_stats = self.cluster_index.get_stats()
        
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
            'annoy_index': annoy_stats,
            'memory_stats': {
                'hot_memories': len(self.hot_memories),
                'normalized_vectors_precomputed': self._normalized_vectors is not None,
                'memory_norms_precomputed': self._precomputed_memory_norms is not None
            }
        }
    
    def clear_all_caches(self):
        """清除所有缓存"""
        with self.vector_cache_lock:
            self.vector_cache.is_valid = False
            self.vector_cache.vectors = None
            self.vector_cache.memory_ids = None
        
        self.similarity_cache.cache.clear()
        self.weight_cache.clear()
        self.cluster_search_cache.clear()
        
        if self.cluster_index:
            self.cluster_index.clear()
        
        self._normalized_vectors = None
        self._precomputed_memory_norms = None
        
        print("[Cache] All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息 - 修复：包含未分配热力信息"""
        stats = self.stats.copy()
        
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
        
        cache_stats = self.get_cache_stats()
        
        if self.cluster_index:
            annoy_stats = self.cluster_index.get_stats()
            stats.update({
                'annoy_index_built': annoy_stats.get('index_built', False),
                'annoy_clusters_indexed': annoy_stats.get('num_clusters', 0),
                'annoy_query_count': annoy_stats.get('query_count', 0),
                'annoy_avg_query_time_ms': annoy_stats.get('avg_query_time', 0) * 1000,
                'annoy_hit_rate': annoy_stats.get('hit_rate', 0),
                'annoy_fallback_searches': annoy_stats.get('fallback_searches', 0),
                'annoy_build_count': annoy_stats.get('build_count', 0)
            })
        
        # 历史记录统计
        history_stats = self.history_manager.get_history_stats()
        stats.update({
            'history_records': history_stats.get('memory_records', 0),
            'history_file_records': history_stats.get('file_records', 0),
            'history_turn_range': history_stats.get('turn_range', (0, 0)),
            'history_loaded_blocks': history_stats.get('loaded_blocks', 0),
            'history_lru_cache_size': history_stats.get('lru_cache_size', 0)
        })
        
        stats.update({
            'memory_addition_count': self.memory_addition_count,
            'operation_count': self.operation_count,
            'memory_additions_since_last_centroid_update': self.memory_additions_since_last_centroid_update,
            'heat_pool': self.heat_pool,
            'unallocated_heat': self.unallocated_heat,
            'memory_heat_total': sum(m.heat for m in self.hot_memories.values()),
            'heat_total_allocated': self.heat_pool + self.unallocated_heat + sum(m.heat for m in self.hot_memories.values()),
            'heat_total_expected': self.config.TOTAL_HEAT,
            'heat_pool_percent': (self.heat_pool / self.config.TOTAL_HEAT * 100) if self.config.TOTAL_HEAT > 0 else 0,
            'unallocated_percent': (self.unallocated_heat / self.config.TOTAL_HEAT * 100) if self.config.TOTAL_HEAT > 0 else 0,
            'memory_heat_percent': ((self.config.TOTAL_HEAT - self.heat_pool - self.unallocated_heat) / self.config.TOTAL_HEAT * 100) if self.config.TOTAL_HEAT > 0 else 0,
            'heat_pool_capacity': self.config.INITIAL_HEAT_POOL,
            'heat_pool_threshold': self.config.INITIAL_HEAT_POOL * self.config.HEAT_POOL_RECYCLE_THRESHOLD,
            'needs_recycle': self.heat_pool < (self.config.INITIAL_HEAT_POOL * self.config.HEAT_POOL_RECYCLE_THRESHOLD),
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
            'annoy_config': {
                'available': ANNOY_AVAILABLE,
                'n_trees': self.config.ANNOY_N_TREES,
                'metric': self.config.ANNOY_METRIC,
                'rebuild_threshold': self.config.ANNOY_REBUILD_THRESHOLD
            },
            'cache_stats': cache_stats
        })
        return stats
    
    def cleanup(self):
        """清理资源"""
        print(f"\n[Memory System] Cleaning up memory module (Final turn: {self.current_turn})...")
        
        self._perform_maintenance_tasks()
        
        if self.memory_addition_count > 0:
            self._create_checkpoint()
        
        cache_stats = self.get_cache_stats()
        print(f"[Cache] Final cache statistics:")
        print(f"  Vector cache: {cache_stats['vector_cache']['size']} vectors")
        print(f"  Similarity cache: {cache_stats['similarity_cache']['size']} entries")
        print(f"  Weight cache: {cache_stats['weight_cache']['size']} entries")
        print(f"  Cluster search cache: {cache_stats['cluster_search_cache']['size']} entries")
        
        if self.cluster_index:
            annoy_stats = cache_stats.get('annoy_index', {})
            print(f"  Annoy index: {annoy_stats.get('num_clusters', 0)} clusters, "
                  f"{annoy_stats.get('query_count', 0)} queries, "
                  f"{annoy_stats.get('hit_rate', 0):.1%} hit rate")
        
        # 历史记录统计
        print(f"[History Manager] Final statistics:")
        history_stats = self.history_manager.get_history_stats()
        for key, value in history_stats.items():
            print(f"  {key}: {value}")
        
        self.background_executor.shutdown(wait=True)
        
        self.conn.close()
        
        print("[Memory System] Cleanup completed")
        
        print("\n[Memory System] Final statistics:")
        stats = self.get_stats()
        
        # 热力系统摘要
        print(f"\n  Heat System Summary:")
        print(f"    Total heat: {self.config.TOTAL_HEAT:,}")
        print(f"    Heat pool: {self.heat_pool:,} ({self.heat_pool/self.config.TOTAL_HEAT*100:.1f}%)")
        print(f"    Unallocated heat: {self.unallocated_heat:,} ({self.unallocated_heat/self.config.TOTAL_HEAT*100:.1f}%)")
        print(f"    Memory heat: {sum(m.heat for m in self.hot_memories.values()):,} "
              f"({sum(m.heat for m in self.hot_memories.values())/self.config.TOTAL_HEAT*100:.1f}%)")
        print(f"    Pool threshold: {self.config.INITIAL_HEAT_POOL * self.config.HEAT_POOL_RECYCLE_THRESHOLD:,}")
        
        for key, value in stats.items():
            if isinstance(value, (int, float)) and value > 0 and key not in ['cache_stats', 'weight_config', 
                                                                            'heat_distribution_config', 
                                                                            'layered_search_config', 
                                                                            'annoy_config']:
                print(f"  {key}: {value}")

# 示例用法
if __name__ == "__main__":
    print("="*80)
    print("Memory Module with History Manager and unified tools")
    print("="*80)
    
    # 示例：创建内存模块
    memory_module = MemoryModule()
    
    # 示例：添加记忆
    memory_id = memory_module.add_memory(
        user_input="这是一个测试用户输入",
        ai_response="这是一个测试AI回答"
    )
    print(f"Added memory with ID: {memory_id}")
    
    # 示例：通过turn查找记忆
    current_turn = memory_module.current_turn
    memory = memory_module.get_memory_by_turn(current_turn)
    if memory:
        print(f"Found memory by turn {current_turn}: {memory.summary}")
    
    # 示例：通过memory_id查找turn
    turn = memory_module.get_turn_by_memory_id(memory_id)
    if turn:
        print(f"Memory {memory_id} was created at turn {turn}")
    
    # 示例：搜索历史记录
    search_results = memory_module.search_history_by_keyword("测试", max_results=5)
    print(f"Found {len(search_results)} history records containing '测试'")
    
    # 示例：获取统计信息
    stats = memory_module.get_stats()
    print(f"Total memories: {stats.get('hot_memories_count', 0)} hot, {stats.get('sleeping_memories_count', 0)} sleeping")
    print(f"History records: {stats.get('history_records', 0)} in memory, {stats.get('history_file_records', 0)} in file")
