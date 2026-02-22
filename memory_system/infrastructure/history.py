from pathlib import Path
import threading
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from ..models import MemoryItem
from ..utils import blob_to_vector
import json


class HistoryManager:
    """历史记录管理器，用于维护 created_turn 到记忆的映射"""

    def __init__(self, memory_module, history_file: str = "./memory/history.txt",
                 max_memory_records: int = 10000, max_disk_records: int = 1000000):
        self.memory_module = memory_module
        self.history_file = Path(history_file)
        self.max_memory_records = max_memory_records
        self.max_disk_records = max_disk_records
        self.embedding_dim = memory_module.embedding_dim

        # 内存中的映射：created_turn -> memory_id
        self.turn_to_memory_id: Dict[int, str] = {}

        # 最近使用的映射：memory_id -> created_turn
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

        self._init_history_file()
        self._load_recent_history()

    def _init_history_file(self):
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self.history_file.write_text("")
        # 索引文件
        index_file = self.history_file.with_suffix('.idx')
        if not index_file.exists():
            index_file.write_text("")

    def _load_recent_history(self):
        try:
            if not self.history_file.exists():
                return
            with open(self.history_file, 'r', encoding='utf-8') as f:
                lines = deque(f, maxlen=self.max_memory_records * 2)
            loaded_count = 0
            for line in lines:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                parts = line.split(',', 2)
                if len(parts) < 2:
                    continue
                try:
                    created_turn = int(parts[0])
                    memory_id = parts[1]
                    self.turn_to_memory_id[created_turn] = memory_id
                    self.memory_id_to_turn[memory_id] = created_turn
                    loaded_count += 1
                    if len(self.turn_to_memory_id) >= self.max_memory_records:
                        break
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f"[History Manager] Error loading history: {e}")

    def add_history_record(self, created_turn: int, memory_id: str, content_preview: str = ""):
        self.turn_to_memory_id[created_turn] = memory_id
        self.memory_id_to_turn[memory_id] = created_turn
        with self.lru_lock:
            if (created_turn, memory_id) in self.lru_cache:
                self.lru_cache.remove((created_turn, memory_id))
            self.lru_cache.append((created_turn, memory_id))
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                content_preview_clean = content_preview.replace('\n', ' ').replace('\r', '')[:200]
                f.write(f"{created_turn},{memory_id},{content_preview_clean}\n")
        except Exception as e:
            print(f"[History Manager] Error writing to history file: {e}")
        if len(self.turn_to_memory_id) >= self.compression_threshold:
            self._compress_memory_storage()

    def get_memory_by_turn(self, created_turn: int) -> Optional[MemoryItem]:
        """根据轮数返回原始对话记忆"""
        # 1. 先检查内存映射
        memory_id = self.turn_to_memory_id.get(created_turn)
        if memory_id:
            with self.lru_lock:
                if (created_turn, memory_id) in self.lru_cache:
                    self.lru_cache.remove((created_turn, memory_id))
                self.lru_cache.append((created_turn, memory_id))
            memory = self._get_memory_from_module(memory_id)
            if memory:
                return memory
        
        # 2. 从文件加载
        memory = self._load_original_dialogue_from_file(created_turn)
        if memory:
            self.turn_to_memory_id[created_turn] = memory.id
            self.memory_id_to_turn[memory.id] = created_turn
            with self.lru_lock:
                self.lru_cache.append((created_turn, memory.id))
        
        return memory

    def get_turn_by_memory_id(self, memory_id: str) -> Optional[int]:
        if memory_id in self.memory_id_to_turn:
            created_turn = self.memory_id_to_turn[memory_id]
            with self.lru_lock:
                if (created_turn, memory_id) in self.lru_cache:
                    self.lru_cache.remove((created_turn, memory_id))
                self.lru_cache.append((created_turn, memory_id))
            return created_turn
        return None

    def get_memories_by_turn_range(self, start_turn: int, end_turn: int) -> List[MemoryItem]:
        """获取指定轮数范围内的原始对话记忆
        
        Args:
            start_turn: 起始轮数（包含）
            end_turn: 结束轮数（包含）
        
        Returns:
            原始对话记忆列表，按创建轮数升序排序
        """
        memories = []
        memory_ids = []
        
        # 1. 首先从内存映射中获取记忆ID
        for turn in range(start_turn, end_turn + 1):
            if turn in self.turn_to_memory_id:
                memory_ids.append(self.turn_to_memory_id[turn])
        
        # 2. 尝试从热区获取这些记忆
        for memory_id in memory_ids:
            memory = self._get_memory_from_module(memory_id)
            if memory:
                memories.append(memory)
        
        # 3. 如果找到的记忆数量不足，从历史文件中补充
        if len(memories) < (end_turn - start_turn + 1):
            additional = self._load_original_dialogues_from_file_range(start_turn, end_turn)
            memories.extend(additional)
        
        # 4. 去重并排序
        seen_ids = set()
        unique_memories = []
        for memory in memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                unique_memories.append(memory)
        
        unique_memories.sort(key=lambda m: m.created_turn)
        return unique_memories

    def _get_memory_from_module(self, memory_id: str) -> Optional[MemoryItem]:
        # 首先检查热区
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
        try:
            block_start = (created_turn // self.block_size) * self.block_size
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
                        if block_start <= turn <= block_start + self.block_size - 1:
                            self.turn_to_memory_id[turn] = memory_id
                            self.memory_id_to_turn[memory_id] = turn
                            if turn == created_turn:
                                return memory_id
                    except (ValueError, IndexError):
                        continue
            self.loaded_blocks.add(block_start)
        except Exception as e:
            print(f"[History Manager] Error loading from file: {e}")
        return None
    
    def _load_original_dialogue_from_file(self, turn: int) -> Optional[MemoryItem]:
        """从历史文件加载单个原始对话"""
        try:
            if not self.history_file.exists():
                return None
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',', 2)
                    if len(parts) < 3:
                        continue
                    try:
                        file_turn = int(parts[0])
                        if file_turn == turn:
                            user_input = parts[1].replace('\\n', '\n')
                            ai_response = parts[2].replace('\\n', '\n')
                            
                            return MemoryItem(
                                id=f"dialogue_{turn}",
                                vector=np.zeros(self.embedding_dim, dtype=np.float32),
                                user_input=user_input,
                                ai_response=ai_response,
                                summary=user_input[:100],
                                heat=0,
                                created_turn=turn,
                                last_interaction_turn=turn,
                                access_count=0,
                                is_hot=False,
                                is_sleeping=False,
                                cluster_id=None,
                                metadata={},
                                version=1,
                                update_count=0,
                                parent_turn=None
                            )
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"[History Manager] Error loading original dialogue from file: {e}")
        return None

    def _load_memories_from_file_range(self, start_turn: int, end_turn: int) -> List[MemoryItem]:
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
                            self.turn_to_memory_id[turn] = memory_id
                            self.memory_id_to_turn[memory_id] = turn
                            if len(self.turn_to_memory_id) >= self.max_memory_records:
                                break
                    except (ValueError, IndexError):
                        continue
            for memory_id in memory_ids_found:
                memory = self._get_memory_from_module(memory_id)
                if memory:
                    memories.append(memory)
        except Exception as e:
            print(f"[History Manager] Error loading range from file: {e}")
        return memories
    
    def _load_original_dialogues_from_file_range(self, start_turn: int, end_turn: int) -> List[MemoryItem]:
        """从历史文件加载指定轮数范围内的原始对话"""
        dialogues = []
        
        try:
            if not self.history_file.exists():
                return dialogues
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析行：turn,user_input,ai_response
                    parts = line.split(',', 2)
                    if len(parts) < 3:
                        continue
                    
                    try:
                        turn = int(parts[0])
                        if start_turn <= turn <= end_turn:
                            user_input = parts[1].replace('\\n', '\n')
                            ai_response = parts[2].replace('\\n', '\n')
                            
                            # 构造原始对话记忆项（不含向量）
                            memory = MemoryItem(
                                id=f"dialogue_{turn}",
                                vector=np.zeros(self.memory_module.embedding_dim, dtype=np.float32),
                                user_input=user_input,
                                ai_response=ai_response,
                                summary=user_input[:100],
                                heat=0,
                                created_turn=turn,
                                last_interaction_turn=turn,
                                access_count=0,
                                is_hot=False,
                                is_sleeping=False,
                                cluster_id=None,
                                metadata={},
                                version=1,
                                update_count=0,
                                parent_turn=None  # 这是原始对话
                            )
                            dialogues.append(memory)
                            
                            # 更新内存映射
                            self.turn_to_memory_id[turn] = memory.id
                            self.memory_id_to_turn[memory.id] = turn
                            
                    except (ValueError, IndexError) as e:
                        continue
                    
                    # 如果已加载足够多，提前退出（可选）
                    if len(dialogues) >= (end_turn - start_turn + 1) * 2:
                        break
                        
        except Exception as e:
            print(f"[History Manager] Error loading original dialogues from file: {e}")
        
        return dialogues

    def _compress_memory_storage(self):
        current_turn = self.memory_module.current_turn
        if current_turn - self.last_compression_turn < 100:
            return
        with self.lru_lock:
            keep_records = set(self.lru_cache)
            new_turn_to_memory_id = {}
            new_memory_id_to_turn = {}
            for created_turn, memory_id in keep_records:
                new_turn_to_memory_id[created_turn] = memory_id
                new_memory_id_to_turn[memory_id] = created_turn
            self.turn_to_memory_id = new_turn_to_memory_id
            self.memory_id_to_turn = new_memory_id_to_turn
            self.lru_cache = deque(self.lru_cache, maxlen=self.max_memory_records)
        self.last_compression_turn = current_turn

    def search_by_content_keyword(self, keyword: str, max_results: int = 50) -> List[Tuple[int, str, str]]:
        results = []
        keyword_lower = keyword.lower()
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
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
        try:
            file_lines = 0
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    file_lines = sum(1 for _ in f)
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
        current_turn = self.memory_module.current_turn
        cutoff_turn = current_turn - max_age_turns
        turns_to_remove = [turn for turn in self.turn_to_memory_id.keys() if turn < cutoff_turn]
        for turn in turns_to_remove:
            memory_id = self.turn_to_memory_id.pop(turn, None)
            if memory_id and memory_id in self.memory_id_to_turn:
                del self.memory_id_to_turn[memory_id]
        with self.lru_lock:
            self.lru_cache = deque(
                [(t, mid) for t, mid in self.lru_cache if t >= cutoff_turn],
                maxlen=self.max_memory_records
            )

    def rebuild_index(self):
        self.turn_to_memory_id.clear()
        self.memory_id_to_turn.clear()
        self.loaded_blocks.clear()
        with self.lru_lock:
            self.lru_cache.clear()
        self._load_recent_history()
        
    def add_original_dialogue(self, turn: int, user_input: str, ai_response: str):
        """将原始对话记录到历史文件"""
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                # 转义换行符，避免破坏 CSV 格式
                user_escaped = user_input.replace('\n', '\\n')
                ai_escaped = ai_response.replace('\n', '\\n')
                f.write(f"{turn},{user_escaped},{ai_escaped}\n")
        except Exception as e:
            print(f"[History Manager] Error writing original dialogue: {e}")