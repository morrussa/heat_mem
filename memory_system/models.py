import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Set
import threading


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
    
    parent_turn: Optional[int] = None

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
        # 新增：处理 parent_turn（如果缺失则为 None）
        if 'parent_turn' not in data:
            data['parent_turn'] = None
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


@dataclass
class WeightedMemoryResult:
    memory: MemoryItem
    base_similarity: float
    relative_heat_weight: float
    access_frequency_weight: float
    recency_weight: float
    final_score: float
    ranking_position: int


@dataclass
class LayeredSearchResult:
    layer_name: str
    similarity_range: Tuple[float, float]
    results: List[WeightedMemoryResult]
    achieved_count: int
    target_count: int
    avg_similarity: float
    avg_final_score: float


@dataclass
class VectorCache:
    vectors: np.ndarray = None
    memory_ids: List[str] = None
    last_updated: float = 0
    is_valid: bool = False
    
# @dataclass
# class PendingHeat:
#     cluster_id: str
#     pending_heat: int
#     version: int = 1
#     last_updated_turn: int = 0

@dataclass
class WaypointEdge:
    source_id: str
    target_id: str
    weight: float
    created_turn: int
    last_updated_turn: int

@dataclass
class PendingHeatUnit:
    id: str                          # 新记忆的ID（也是主键）
    vector: np.ndarray                # 新记忆的向量
    pending_heat: int                 # 暂存的热力值
    created_turn: int                 # 创建时的轮数
    status: str = "pending"           # 状态：pending / processing / done
    version: int = 1