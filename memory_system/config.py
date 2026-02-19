# =============== 配置常量 ===============
class Config:
    
    WAYPOINT_TABLE = "waypoint_edges"
    WAYPOINT_MIN_WEIGHT = 0.1           # 边权重低于此值则删除
    WAYPOINT_DEFAULT_WEIGHT = 0.5       # 新建边的默认权重
    WAYPOINT_DECAY_FACTOR = 0.99        # 每轮衰减因子
    WAYPOINT_MIN_DECAYED_WEIGHT = 0.01  # 衰减后低于此值的边不返回
    WAYPOINT_REINFORCE_DELTA = 0.1      # 每次强化增量
    WAYPOINT_COOCCUR_INCREMENT = 0.2    # 每次共现增量
    WAYPOINT_SEMANTIC_LOW = 0.85        # 语义相似度下界
    WAYPOINT_SEMANTIC_HIGH = 0.95       # 语义相似度上界
    WAYPOINT_SEMANTIC_INIT_WEIGHT = 0.3 # 语义弱边的初始权重
    WAYPOINT_MAX_HOPS = 1                # 联想最大深度（1或2）
    WAYPOINT_MAX_RESULTS_PER_SEED = 3    # 每个种子最多联想结果
    WAYPOINT_GLOBAL_MAX = 5              # 总联想结果上限
    
    COLD_DOMINANT_HOT_COUNT_THRESHOLD = 3  # 热区记忆少于这个数认为冷主导
    COLD_DOMINANT_COLD_COUNT_MIN = 1        # 冷区记忆至少要有这个数

    WAKEUP_MEMORIES_PER_CLUSTER = 5         # 每次唤醒每个簇最多唤醒的记忆数
    PENDING_HEAT_PROCESS_INTERVAL = 100     # 每多少轮处理一次暂存热力
    
    RECENCY_BONUS_MAX = 0.2           # 最大奖励值
    RECENCY_BONUS_WINDOW = 40          # 考虑最近多少轮内的访问
    RECENCY_BONUS_SATURATION = 5       # 奖励饱和所需的访问次数（若次数≥5则取最大值）
    
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
    
    # 在 Config 类中添加
    PENDING_HEAT_TABLE = "pending_heat"  # 暂存热力表名
    PENDING_HEAT_CACHE_TTL = 100  # 暂存热力缓存 TTL（轮数）
    
    


# =============== 枚举和常量 ===============
from enum import Enum

class OperationType(Enum):
    MEMORY_HEAT_UPDATE = "memory_heat_update"
    CLUSTER_HEAT_UPDATE = "cluster_heat_update"
    MEMORY_TO_COLD = "memory_to_cold"
    MEMORY_TO_HOT = "memory_to_hot"
    CLUSTER_CREATE = "cluster_create"
    CLUSTER_UPDATE = "cluster_update"
    CLUSTER_DELETE = "cluster_delete"
    PENDING_HEAT_UPDATE = "pending_heat_update"
    POOL_HEAT_UPDATE = "pool_heat_update"

class ConsistencyLevel(Enum):
    EVENTUAL = "eventual"
    IMMEDIATE = "immediate"
    STRONG = "strong"
    
