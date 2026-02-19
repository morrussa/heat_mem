import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Set
import hashlib


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