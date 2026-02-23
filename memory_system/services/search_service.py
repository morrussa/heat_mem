import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from ..config import Config
from ..models import WeightedMemoryResult, LayeredSearchResult, MemoryItem
from ..utils import compute_cosine_similarity, compute_batch_similarities


class SearchService:
    def __init__(self, memory_module):
        self.memory_module = memory_module
        self.config: Config = memory_module.config

    def _get_relative_heat_weight(self, memory_item, cluster_total_heat: int) -> float:
        if cluster_total_heat <= 0:
            return 1.0
        relative_heat = memory_item.heat / cluster_total_heat
        weight = relative_heat ** self.config.RELATIVE_HEAT_WEIGHT_POWER
        return max(0.5, min(1.0, weight))

    def _get_access_frequency_weight(self, memory_id: str, memory_item) -> float:
        with self.memory_module.frequency_stats_lock:
            if memory_id not in self.memory_module.access_frequency_stats:
                return 1.0
            stats = self.memory_module.access_frequency_stats[memory_id]
            access_count = stats['count']
            recent_interactions = [turn for turn in stats['recent_interactions']
                                  if self.memory_module.current_turn - turn < 1000]
            recent_count = len(recent_interactions)
            total_factor = min(1.0, self.config.ACCESS_FREQUENCY_DISCOUNT_THRESHOLD / max(1, access_count))
            recent_factor = min(1.0, self.config.ACCESS_FREQUENCY_DISCOUNT_THRESHOLD / max(1, recent_count))
            weight = 0.3 * total_factor + 0.7 * recent_factor
            if memory_item.is_sleeping:
                weight *= 0.5
            return max(0.1, weight)

    def _get_recency_bonus(self, memory_id: str) -> float:
        """根据最近窗口内的访问次数计算新鲜度奖励（0～RECENCY_BONUS_MAX）"""
        with self.memory_module.frequency_stats_lock:
            stats = self.memory_module.access_frequency_stats.get(memory_id)
            if not stats:
                return 0.0

            recent = stats.get('recent_interactions', [])
            if not recent:
                return 0.0

            window_start = self.memory_module.current_turn - self.config.RECENCY_BONUS_WINDOW
            count_in_window = sum(1 for turn in recent if turn >= window_start)

            if count_in_window == 0:
                return 0.0

            bonus = self.config.RECENCY_BONUS_MAX * min(
                1.0,
                count_in_window / self.config.RECENCY_BONUS_SATURATION
            )
            return min(self.config.RECENCY_BONUS_MAX, bonus)

    def _get_cached_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        cached = self.memory_module.cache_manager.similarity_cache.get(query_vector)
        if cached is not None:
            self.memory_module.stats['similarity_cache_hits'] += 1
            return cached
        self.memory_module.stats['similarity_cache_misses'] += 1
        similarities = self._compute_all_similarities_vectorized(query_vector)
        self.memory_module.cache_manager.similarity_cache.put(query_vector, similarities)
        return similarities

    def _compute_all_similarities_vectorized(self, query_vector: np.ndarray) -> np.ndarray:
        self.memory_module.cache_manager.ensure_vector_cache(self.memory_module.hot_memories)
        vectors = self.memory_module.cache_manager.vector_cache.vectors
        if vectors.shape[0] == 0:
            return np.array([])
        return compute_batch_similarities(query_vector, vectors)

    def _ensure_weight_cache(self):
        """确保权重缓存有效，只缓存相对热力权重和访问频率权重"""
        current_turn = self.memory_module.current_turn
        if (current_turn - self.memory_module.cache_manager.weight_cache_turn > 100 or
            len(self.memory_module.cache_manager.weight_cache) != len(self.memory_module.hot_memories)):
            with self.memory_module.frequency_stats_lock:
                self.memory_module.cache_manager.weight_cache.clear()
                for memory_id, memory in self.memory_module.hot_memories.items():
                    cluster_total_heat = self.memory_module.clusters[memory.cluster_id].total_heat if memory.cluster_id in self.memory_module.clusters else 1
                    self.memory_module.cache_manager.weight_cache[memory_id] = {
                        'relative_heat_weight': self._get_relative_heat_weight(memory, cluster_total_heat),
                        'access_frequency_weight': self._get_access_frequency_weight(memory_id, memory),
                        'heat': memory.heat,
                        'last_updated_turn': current_turn
                    }
            self.memory_module.cache_manager.weight_cache_turn = current_turn

    def search_layered_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                               max_total_results: int = None,
                               config_override: Dict = None) -> Dict[str, LayeredSearchResult]:
        if not self.config.LAYERED_SEARCH_ENABLED:
            warnings.warn("Layered search is disabled. Using default search.")
            return self._fallback_search(query_text, query_vector, max_total_results)

        if query_vector is None and query_text is not None:
            query_vector = self.memory_module._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")

        config = config_override or self.config.LAYERED_SEARCH_CONFIG
        if max_total_results is None:
            max_total_results = self.config.LAYERED_SEARCH_MAX_TOTAL_RESULTS

        self.memory_module.stats['layered_searches'] += 1
        self.memory_module.stats['vectorized_searches'] += 1

        start_time = time.time()
        similarities = self._get_cached_similarities(query_vector)
        sim_compute_time = time.time() - start_time

        self._ensure_weight_cache()

        memory_ids = self.memory_module.cache_manager.vector_cache.memory_ids
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
                memory = self.memory_module.hot_memories[memory_id]
                if memory.heat < min_heat:
                    continue
                cached_weights = self.memory_module.cache_manager.weight_cache.get(memory_id, {})
                if not cached_weights:
                    cluster_total_heat = self.memory_module.clusters[memory.cluster_id].total_heat if memory.cluster_id in self.memory_module.clusters else 1
                    relative_heat_weight = self._get_relative_heat_weight(memory, cluster_total_heat)
                    access_frequency_weight = self._get_access_frequency_weight(memory_id, memory)
                else:
                    relative_heat_weight = cached_weights['relative_heat_weight']
                    access_frequency_weight = cached_weights['access_frequency_weight']

                heat_weight_factor = layer_config.get("heat_weight_factor", 1.0)
                frequency_weight_factor = layer_config.get("frequency_weight_factor", 1.0)
                recency_weight_factor = layer_config.get("recency_weight_factor", 1.0)
                base_score_factor = layer_config.get("base_score_factor", 1.0)

                adj_relative_heat_weight = relative_heat_weight * heat_weight_factor
                adj_access_frequency_weight = access_frequency_weight * frequency_weight_factor
                recency_bonus = self._get_recency_bonus(memory_id) * recency_weight_factor

                # 对热度和访问频率计算几何平均
                weights = [adj_relative_heat_weight, adj_access_frequency_weight]
                weights_nonzero = [max(0.0001, w) for w in weights]
                geometric_mean = np.exp(np.mean(np.log(weights_nonzero)))
                
                # 最终得分 = 相似度 × 几何平均 + 新鲜度奖励
                final_score = similarities[idx] * geometric_mean + recency_bonus

                result = WeightedMemoryResult(
                    memory=memory,
                    base_similarity=similarities[idx],
                    relative_heat_weight=adj_relative_heat_weight,
                    access_frequency_weight=adj_access_frequency_weight,
                    recency_weight=recency_bonus,  # 使用新鲜度奖励作为recency_weight
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
        if query_vector is None and query_text is not None:
            query_vector = self.memory_module._get_embedding(query_text)
        if max_total_results is None:
            max_total_results = 8
        all_results = self.search_similar_memories(
            query_vector=query_vector,
            max_results=max_total_results,
            use_weighting=True
        )
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
        layered_results = {}
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
        layered_results = self.search_layered_memories(
            query_text=query_text,
            query_vector=query_vector
        )
        if not flatten_results:
            return layered_results
        flattened = []
        for layer_name in ["layer_3", "layer_2", "layer_1"]:
            if layer_name in layered_results:
                flattened.extend(layered_results[layer_name].results)
        return flattened

    def search_within_cluster(self, query_text: str = None, query_vector: np.ndarray = None,
                             cluster_id: str = None, max_results: int = None) -> List[WeightedMemoryResult]:
        if max_results is None:
            max_results = self.config.CLUSTER_SEARCH_MAX_RESULTS
        if query_vector is None and query_text is not None:
            query_vector = self.memory_module._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")

        if cluster_id is None:
            if self.memory_module.cluster_service.cluster_index and len(self.memory_module.clusters) >= 3:
                results = self.memory_module.cluster_service.cluster_index.find_nearest_clusters(
                    query_vector,
                    n=1,
                    search_k=self.config.ANNOY_SEARCH_K
                )
                if results:
                    cluster_id = results[0][0]
            else:
                best_cluster_id, best_similarity = self.memory_module.cluster_service._find_best_cluster_linear(query_vector)
                cluster_id = best_cluster_id

        if cluster_id is None or cluster_id not in self.memory_module.clusters:
            return []

        cached_results = self.memory_module.cache_manager.cluster_search_cache.get(cluster_id, query_vector, self.memory_module.current_turn)
        if cached_results is not None:
            self.memory_module.stats['cache_hits'] += 1
            return cached_results[:max_results]

        self.memory_module.stats['cache_misses'] += 1
        self.memory_module.stats['cluster_searches'] += 1

        cluster = self.memory_module.clusters[cluster_id]
        cluster_total_heat = cluster.total_heat
        weighted_results = []
        cluster_memory_ids = set()
        with cluster.lock:
            cluster_memory_ids.update(cluster.memory_ids)
        for memory_id, memory in self.memory_module.hot_memories.items():
            if memory.cluster_id == cluster_id:
                cluster_memory_ids.add(memory_id)
        for memory_id in cluster_memory_ids:
            memory = self.memory_module.hot_memories.get(memory_id)
            if memory is None or memory.is_sleeping:
                continue
            base_similarity = compute_cosine_similarity(query_vector, memory.vector)
            if base_similarity < self.config.SIMILARITY_THRESHOLD:
                continue
            relative_heat_weight = self._get_relative_heat_weight(memory, cluster_total_heat)
            access_frequency_weight = self._get_access_frequency_weight(memory_id, memory)
            recency_bonus = self._get_recency_bonus(memory_id)
            
            weights = [relative_heat_weight, access_frequency_weight]
            weights_nonzero = [max(0.0001, w) for w in weights]
            geometric_mean = np.exp(np.mean(np.log(weights_nonzero)))
            
            final_score = base_similarity * geometric_mean + recency_bonus
            result = WeightedMemoryResult(
                memory=memory,
                base_similarity=base_similarity,
                relative_heat_weight=relative_heat_weight,
                access_frequency_weight=access_frequency_weight,
                recency_weight=recency_bonus,  # 使用新鲜度奖励作为recency_weight
                final_score=final_score,
                ranking_position=0
            )
            weighted_results.append(result)

        weighted_results.sort(key=lambda x: x.final_score, reverse=True)
        for i, result in enumerate(weighted_results[:max_results]):
            result.ranking_position = i + 1
        final_results = weighted_results[:max_results]
        self.memory_module.stats['weight_adjustments'] += len(final_results)
        high_freq_count = sum(1 for r in final_results if r.access_frequency_weight < 0.5)
        self.memory_module.stats['high_frequency_memories'] += high_freq_count
        if final_results:
            self.memory_module.cache_manager.cluster_search_cache.put(cluster_id, query_vector, final_results, self.memory_module.current_turn)
        return final_results

    def search_similar_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                               max_results: int = 10, use_weighting: bool = True) -> List[WeightedMemoryResult]:
        if query_vector is None and query_text is not None:
            query_vector = self.memory_module._get_embedding(query_text)
        elif query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")

        all_results = []
        if use_weighting:
            for cluster_id in self.memory_module.clusters.keys():
                cluster_results = self.search_within_cluster(
                    query_vector=query_vector,
                    cluster_id=cluster_id,
                    max_results=max_results // 2
                )
                all_results.extend(cluster_results)
        else:
            for memory_id, memory in self.memory_module.hot_memories.items():
                if memory.is_sleeping:
                    continue
                similarity = compute_cosine_similarity(query_vector, memory.vector)
                if similarity >= self.config.SIMILARITY_THRESHOLD:
                    recency_bonus = self._get_recency_bonus(memory_id)
                    result = WeightedMemoryResult(
                        memory=memory,
                        base_similarity=similarity,
                        relative_heat_weight=1.0,
                        access_frequency_weight=1.0,
                        recency_weight=recency_bonus,
                        final_score=similarity + recency_bonus,  # 简化的评分
                        ranking_position=0
                    )
                    all_results.append(result)
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        for i, result in enumerate(all_results[:max_results]):
            result.ranking_position = i + 1
        return all_results[:max_results]
    
    def search_original_memories(self, query_text: str = None, query_vector: np.ndarray = None,
                                 max_results: int = 10) -> List[Tuple[MemoryItem, float]]:
        """
        搜索原子事实，聚合返回原始对话，并按得分排序。
        不再使用联想扩散（Waypoint），直接返回种子记忆。
        """
        if query_vector is None and query_text is not None:
            query_vector = self.memory_module._get_embedding(query_text)
        
        # 1. 搜索原子事实（放宽数量限制）
        atomic_results = self.search_similar_memories(
            query_vector=query_vector,
            max_results=max_results * 3,
            use_weighting=True
        )
        
        # 2. 按 parent_turn 分组，保留每个原始对话的最高得分（种子记忆）
        turn_to_score = {}
        for res in atomic_results:
            mem = res.memory
            # 获取所有关联的原始对话轮次
            parent_turns = []
            if mem.metadata and "parent_turns" in mem.metadata:
                parent_turns = mem.metadata["parent_turns"]
            elif mem.parent_turn is not None:
                parent_turns = [mem.parent_turn]
            
            for pt in parent_turns:
                if pt not in turn_to_score or res.final_score > turn_to_score[pt]:
                    turn_to_score[pt] = res.final_score
        
        # 根据轮次获取原始对话内容，构建种子记忆列表
        seed_memories = []  # List[Tuple[MemoryItem, float]]
        for parent_turn, score in turn_to_score.items():
            # 注意：原代码使用 history_manager，如果已改为 dialogue_manager 请相应调整
            original = self.memory_module.history_manager.get_memory_by_turn(parent_turn)
            if original:
                seed_memories.append((original, score))
        
        # 按得分排序后返回
        seed_memories.sort(key=lambda x: x[1], reverse=True)
        return seed_memories[:max_results]