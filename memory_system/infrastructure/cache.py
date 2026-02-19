import numpy as np
import threading
import time
import hashlib
from typing import Dict, List, Optional, Any
from ..models import WeightedMemoryResult, VectorCache


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


class CacheManager:
    def __init__(self, config):
        self._normalized_vectors = None
        self._precomputed_memory_norms = None
        self.config = config
        self.vector_cache = VectorCache()
        self.vector_cache_lock = threading.RLock()
        self.similarity_cache = SimilarityCache(
            max_size=100,
            ttl_seconds=300
        )
        self.cluster_search_cache = ClusterSearchCache(
            max_size=config.CLUSTER_SEARCH_CACHE_SIZE,
            ttl_turns=config.CLUSTER_SEARCH_CACHE_TTL_TURNS
        )
        # weight_cache 现在只存储相对热度权重和访问频率权重
        self.weight_cache: Dict[str, Dict] = {}
        self.weight_cache_turn = 0
        # 新增：记录上次权重缓存构建时的统计信息，用于判断是否需要重建
        self.last_cache_stats = {
            'hot_memories_count': 0,
            'clusters_count': 0,
            'total_heat': 0
        }

    def invalidate_vector_cache(self, memory_id: str = None):
        with self.vector_cache_lock:
            if memory_id and self.vector_cache.is_valid and self.vector_cache.vectors is not None:
                if memory_id in self.vector_cache.memory_ids:
                    self.vector_cache.is_valid = False
            else:
                self.vector_cache.is_valid = False

    def rebuild_vector_cache(self, hot_memories_dict):
        with self.vector_cache_lock:
            from ..utils import convert_memory_vectors
            memory_ids, vectors = convert_memory_vectors(list(hot_memories_dict.values()))
            if vectors.shape[0] > 0:
                self.vector_cache.vectors = vectors
            else:
                self.vector_cache.vectors = np.zeros((0, self.config.EMBEDDING_DIM), dtype=np.float32)
            self.vector_cache.memory_ids = memory_ids
            self.vector_cache.last_updated = time.time()
            self.vector_cache.is_valid = True
            self._normalized_vectors = None
            self._precomputed_memory_norms = None
            print(f"[Vector Cache] Rebuilt cache with {len(memory_ids)} vectors")

    def ensure_vector_cache(self, hot_memories_dict):
        with self.vector_cache_lock:
            if (self.vector_cache.is_valid and
                self.vector_cache.vectors is not None and
                len(self.vector_cache.memory_ids) == len(hot_memories_dict)):
                return
            self.rebuild_vector_cache(hot_memories_dict)

    def clear_all(self):
        with self.vector_cache_lock:
            self.vector_cache.is_valid = False
            self.vector_cache.vectors = None
            self.vector_cache.memory_ids = None
        self.similarity_cache.cache.clear()
        self.weight_cache.clear()
        self.cluster_search_cache.clear()
        self._normalized_vectors = None
        self._precomputed_memory_norms = None
        self.last_cache_stats = {
            'hot_memories_count': 0,
            'clusters_count': 0,
            'total_heat': 0
        }

    def get_stats(self):
        return {
            'vector_cache': {
                'is_valid': self.vector_cache.is_valid,
                'size': len(self.vector_cache.memory_ids) if self.vector_cache.memory_ids else 0,
                'last_updated': self.vector_cache.last_updated,
                'age_seconds': time.time() - self.vector_cache.last_updated if self.vector_cache.last_updated > 0 else 0
            },
            'similarity_cache': self.similarity_cache.get_stats(),
            'weight_cache': {
                'size': len(self.weight_cache),
                'last_updated_turn': self.weight_cache_turn,
                'memory_count': self.last_cache_stats['hot_memories_count']
            },
            'cluster_search_cache': {
                'size': len(self.cluster_search_cache.cache),
                'access_turns_size': len(self.cluster_search_cache.access_turns)
            }
        }