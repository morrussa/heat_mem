import numpy as np
import threading
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from ..config import Config
import time
from ..models import SemanticCluster, MemoryItem
from ..utils import compute_cosine_similarity, vector_to_blob, blob_to_vector, schedule_centroid_update
from ..infrastructure.locking import DistributedLockManager

# 尝试导入Annoy
try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False


class ClusterCentroidIndex:
    def __init__(self, embedding_dim: int, metric: str = 'angular',
                 n_trees: int = 10, rebuild_threshold: int = 50):
        if not ANNOY_AVAILABLE:
            raise ImportError("Annoy is not installed.")
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
            start_time = time.time()
            self._create_new_index()
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
            self.stats['build_count'] += 1

    def find_nearest_clusters(self, vector: np.ndarray, n: int = 5,
                             search_k: int = -1) -> List[Tuple[str, float]]:
        start_time = time.time()
        with self.lock:
            if not self.index_built or len(self.centroid_vectors) == 0:
                self.stats['fallback_searches'] += 1
                return []
            try:
                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
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
                self.stats['avg_query_time'] = self.stats['total_query_time'] / self.stats['query_count']
                if results:
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
                return results
            except Exception as e:
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


class ClusterService:
    def __init__(self, memory_module):
        self.memory_module = memory_module
        self.config: Config = memory_module.config
        self.lock_manager: DistributedLockManager = memory_module.lock_manager
        self.cluster_index = None
        if ANNOY_AVAILABLE:
            try:
                self.cluster_index = ClusterCentroidIndex(
                    embedding_dim=self.config.EMBEDDING_DIM,
                    metric=self.config.ANNOY_METRIC,
                    n_trees=self.config.ANNOY_N_TREES,
                    rebuild_threshold=self.config.ANNOY_REBUILD_THRESHOLD
                )
            except Exception as e:
                print(f"Failed to initialize Annoy index: {e}")
                self.cluster_index = None
    
    def load_cold_memories_from_cluster(self, cluster_id: str, limit: int = 5) -> List[Dict]:
        """从数据库加载指定簇的冷区记忆"""
        try:
            self.memory_module.cursor.execute(f"""
                SELECT * FROM {self.memory_module.config.MEMORY_TABLE}
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
                    'summary': row['summary'],
                    'heat': row['heat'],
                    'created_turn': row['created_turn'],
                    'last_interaction_turn': row['last_interaction_turn'],
                    'access_count': row['access_count'],
                    'metadata': row['metadata'],
                    'parent_turn': row.get('parent_turn')
                })
            return memories
        except Exception as e:
            print(f"[Memory] Error loading cold memories from cluster {cluster_id}: {e}")
            return []

    def _find_best_cluster_annoy(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        if self.cluster_index is None or len(self.memory_module.clusters) < 3:
            return self._find_best_cluster_linear(vector)
        try:
            results = self.cluster_index.find_nearest_clusters(
                vector,
                n=min(5, len(self.memory_module.clusters)),
                search_k=self.config.ANNOY_SEARCH_K
            )
            self.memory_module.stats['annoy_queries'] += 1
            if not results:
                self.memory_module.stats['annoy_fallback_searches'] += 1
                return self._find_best_cluster_linear(vector)
            best_cluster_id, best_similarity = results[0]
            if best_cluster_id in self.memory_module.clusters:
                cluster = self.memory_module.clusters[best_cluster_id]
                if cluster.centroid is not None:
                    actual_similarity = compute_cosine_similarity(vector, cluster.centroid)
                    return best_cluster_id, actual_similarity
            self.memory_module.stats['annoy_fallback_searches'] += 1
            return self._find_best_cluster_linear(vector)
        except Exception as e:
            self.memory_module.stats['annoy_fallback_searches'] += 1
            return self._find_best_cluster_linear(vector)

    def _find_best_cluster_linear(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        best_cluster_id = None
        best_similarity = -1.0
        for cluster_id, centroid in self.memory_module.cluster_vectors.items():
            similarity = compute_cosine_similarity(vector, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id
        return best_cluster_id, best_similarity

    def _update_cluster_index(self, cluster_id: str, centroid: np.ndarray = None,
                             operation: str = 'add'):
        if self.cluster_index is None or not ANNOY_AVAILABLE:
            return
        if centroid is None and cluster_id in self.memory_module.clusters:
            cluster = self.memory_module.clusters[cluster_id]
            centroid = cluster.centroid
        if centroid is not None:
            if operation in ('add', 'update'):
                self.cluster_index.add_cluster(cluster_id, centroid, self.memory_module.current_turn)
            elif operation == 'remove':
                self.cluster_index.remove_cluster(cluster_id)

    def _rebuild_cluster_index(self):
        if self.cluster_index is None:
            return
        if self.cluster_index.changes_since_last_build < self.config.ANNOY_REBUILD_THRESHOLD:
            return
        self.cluster_index.clear()
        for cluster_id, cluster in self.memory_module.clusters.items():
            if cluster.centroid is not None:
                self.cluster_index.add_cluster(cluster_id, cluster.centroid, self.memory_module.current_turn)
        self.cluster_index.build_index(force=True)

    def _assign_to_cluster(self, memory: MemoryItem, vector: np.ndarray) -> str:
        best_cluster_id, best_similarity = self._find_best_cluster_annoy(vector)
        if best_similarity >= self.config.CLUSTER_SIMILARITY_THRESHOLD:
            cluster_id = best_cluster_id
        else:
            cluster_id = f"cluster_{self.memory_module.current_turn}_{hashlib.md5(vector.tobytes()).hexdigest()[:8]}"
            cluster = SemanticCluster(
                id=cluster_id,
                centroid=vector.copy(),
                total_heat=0,
                hot_memory_count=0,
                cold_memory_count=0,
                is_loaded=True,
                size=0,
                last_updated_turn=self.memory_module.current_turn,
                memory_additions_since_last_update=0
            )
            self.memory_module.clusters[cluster_id] = cluster
            self.memory_module.cluster_vectors[cluster_id] = cluster.centroid
            self._update_cluster_index(cluster_id, vector, 'add')
            self.memory_module.stats['clusters'] += 1
            self._save_cluster_to_db(cluster)
        cluster = self.memory_module.clusters[cluster_id]
        with cluster.lock:
            cluster.memory_ids.add(memory.id)
            cluster.size += 1
            cluster.hot_memory_count += 1
            cluster.is_loaded = True
        memory.cluster_id = cluster_id
        return cluster_id

    def _save_cluster_to_db(self, cluster: SemanticCluster):
        self.memory_module.cursor.execute(f"""
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
        self.memory_module.cursor.execute(f"""
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
            self.memory_module.current_turn,
            cluster.memory_additions_since_last_update,
            cluster.id
        ))
        cluster.version += 1

    def _unified_centroid_management(self, cluster_id: str, vector: np.ndarray,
                                   operation: str, memory_id: str = None):
        if cluster_id in self.memory_module.clusters:
            schedule_centroid_update(
                self.memory_module.clusters, cluster_id, vector,
                self.memory_module.clusters_needing_centroid_update,
                add=(operation == 'add')
            )
        if self.cluster_index and ANNOY_AVAILABLE:
            if operation == 'add':
                self.cluster_index.add_cluster(cluster_id, vector, self.memory_module.current_turn)
            elif operation == 'remove':
                self.cluster_index.remove_cluster(cluster_id)
        if operation == 'add':
            self.memory_module.memory_additions_since_last_centroid_update += 1
        self.memory_module._invalidate_related_caches(memory_id, cluster_id)

    def _update_cluster_centroids_batch(self):
        if self.memory_module.memory_additions_since_last_centroid_update < self.config.CENTROID_UPDATE_FREQUENCY:
            return
        self.memory_module.memory_additions_since_last_centroid_update = 0
        clusters_to_update = list(self.memory_module.clusters_needing_centroid_update)
        self.memory_module.clusters_needing_centroid_update.clear()
        if not clusters_to_update:
            for cluster_id, cluster in self.memory_module.clusters.items():
                if cluster.memory_additions_since_last_update > 0:
                    clusters_to_update.append(cluster_id)
        batch_size = self.config.CENTROID_UPDATE_BATCH_SIZE
        for i in range(0, len(clusters_to_update), batch_size):
            batch = clusters_to_update[i:i+batch_size]
            self._update_cluster_centroids(batch)

    def _update_cluster_centroids(self, cluster_ids: List[str]):
        centroid_updates = {}
        for cluster_id in cluster_ids:
            if cluster_id not in self.memory_module.clusters:
                continue
            cluster = self.memory_module.clusters[cluster_id]
            if cluster.memory_additions_since_last_update == 0 and not cluster.pending_centroid_updates:
                continue
            with cluster.lock:
                if cluster.memory_additions_since_last_update >= self.config.CENTROID_FULL_RECALC_THRESHOLD:
                    new_centroid = self._recalculate_cluster_centroid(cluster_id)
                    self.memory_module.stats['full_centroid_recalculations'] += 1
                else:
                    new_centroid = self._incremental_update_cluster_centroid(cluster)
                if new_centroid is not None:
                    cluster.centroid = new_centroid
                    cluster.memory_additions_since_last_update = 0
                    cluster.pending_centroid_updates.clear()
                    cluster.last_updated_turn = self.memory_module.current_turn
                    cluster.version += 1
                    self.memory_module.cluster_vectors[cluster_id] = new_centroid
                    self._update_cluster_index(cluster_id, new_centroid, 'update')
                    centroid_updates[cluster_id] = {
                        'centroid': new_centroid,
                        'turn': self.memory_module.current_turn
                    }
        if centroid_updates:
            with self.memory_module.conn:
                for cluster_id, update_data in centroid_updates.items():
                    self.memory_module.cursor.execute(f"""
                        UPDATE {self.config.CLUSTER_TABLE}
                        SET centroid = ?, last_updated_turn = ?, version = version + 1,
                            memory_additions_since_last_update = 0
                        WHERE id = ?
                    """, (
                        vector_to_blob(update_data['centroid']),
                        update_data['turn'],
                        cluster_id
                    ))
            self.memory_module.stats['centroid_updates'] += len(centroid_updates)
            for cluster_id in cluster_ids:
                self.memory_module.cache_manager.cluster_search_cache.clear(cluster_id)

    def _incremental_update_cluster_centroid(self, cluster: SemanticCluster) -> Optional[np.ndarray]:
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
                        new_centroid = np.zeros(self.config.EMBEDDING_DIM, dtype=np.float32)
                    cluster.size -= 1
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            return new_centroid
        return None

    def _recalculate_cluster_centroid(self, cluster_id: str) -> Optional[np.ndarray]:
        if cluster_id not in self.memory_module.clusters:
            return None
        self.memory_module.cursor.execute(f"""
            SELECT vector FROM {self.config.MEMORY_TABLE}
            WHERE cluster_id = ? AND is_hot = 1
            LIMIT 1000
        """, (cluster_id,))
        rows = self.memory_module.cursor.fetchall()
        if not rows:
            return None
        vectors = [blob_to_vector(row['vector']) for row in rows]
        new_centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        return new_centroid

    def find_best_clusters_for_query(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_vector = self.memory_module._get_embedding(query)
        if self.cluster_index and ANNOY_AVAILABLE:
            try:
                results = self.cluster_index.find_nearest_clusters(
                    query_vector,
                    n=min(top_k, len(self.memory_module.clusters)),
                    search_k=self.config.ANNOY_SEARCH_K
                )
                valid_results = []
                for cluster_id, similarity in results:
                    if cluster_id in self.memory_module.clusters:
                        valid_results.append((cluster_id, similarity))
                if valid_results:
                    return valid_results[:top_k]
            except Exception:
                pass
        # 线性搜索
        results = []
        for cluster_id, cluster in self.memory_module.clusters.items():
            if cluster.centroid is not None:
                similarity = compute_cosine_similarity(query_vector, cluster.centroid)
                results.append((cluster_id, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_cluster_statistics(self, cluster_id: str) -> Dict[str, Any]:
        if cluster_id not in self.memory_module.clusters:
            return {}
        cluster = self.memory_module.clusters[cluster_id]
        memories_in_cluster = []
        for memory_id, memory in self.memory_module.hot_memories.items():
            if memory.cluster_id == cluster_id and not memory.is_sleeping:
                memories_in_cluster.append(memory)
        if not memories_in_cluster:
            return {
                'cluster_id': cluster_id,
                'total_heat': cluster.total_heat,
                'memory_count': 0,
                'heat_distribution': [],
                'frequency_stats': [],
                'current_turn': self.memory_module.current_turn
            }
        heat_values = [m.heat for m in memories_in_cluster]
        total_heat = sum(heat_values)
        heat_distribution = []
        for memory in memories_in_cluster:
            relative_heat = memory.heat / total_heat if total_heat > 0 else 0.0
            access_weight = self.memory_module._get_access_frequency_weight(memory.id, memory)
            heat_distribution.append({
                'memory_id': memory.id,
                'heat': memory.heat,
                'relative_heat': relative_heat,
                'access_count': memory.access_count,
                'access_frequency_weight': access_weight,
                'last_interaction_turn': memory.last_interaction_turn,
                'turns_since_interaction': self.memory_module.current_turn - memory.last_interaction_turn
            })
        heat_distribution.sort(key=lambda x: x['heat'], reverse=True)
        frequency_stats = []
        for memory in memories_in_cluster:
            stats = self.memory_module.access_frequency_stats.get(memory.id, {})
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
            'current_turn': self.memory_module.current_turn,
            'cluster_last_updated_turn': cluster.last_updated_turn,
            'turns_since_cluster_update': self.memory_module.current_turn - cluster.last_updated_turn
        }