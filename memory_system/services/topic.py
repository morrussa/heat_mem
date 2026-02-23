# memory_system/services/topic.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import threading
import json
from pathlib import Path


class TopicSegmenter:
    """话题分割器：基于原始对话的用户输入向量进行话题边界检测和概括
    
    职责：
    1. 接收每一轮的用户输入向量，检测话题边界
    2. 当话题结束时，生成话题摘要
    3. 管理话题摘要的持久化（history.idx）
    
    数据流向：
    - 输入：用户向量（来自main.py）
    - 输出：话题摘要写入 history.idx
    - 依赖：DialogueManager 用于获取对话内容生成摘要
    """
    
    def __init__(self, dialogue_manager, similarity_threshold: float = 0.4,
                 consecutive_low_threshold: int = 2,                 # NEW
                 idx_file_path: Path = None, small_llm_func: Callable = None):
        """初始化话题分割器
        
        Args:
            dialogue_manager: 对话管理器实例，用于获取原始对话内容
            similarity_threshold: 话题边界检测的相似度阈值（低于此值视为可能的新话题）
            consecutive_low_threshold: 连续低于阈值达到此次数后才确认话题切换，默认1（立即切换）
            idx_file_path: 话题摘要文件路径
            small_llm_func: 用于生成摘要的小模型函数
        """
        self.dialogue_manager = dialogue_manager
        self.similarity_threshold = similarity_threshold
        self.consecutive_low_threshold = max(1, consecutive_low_threshold)  # NEW 至少为1
        self.idx_file_path = idx_file_path or Path("./memory/history.idx")
        self.small_llm_func = small_llm_func

        # 核心数据
        self.turn_vectors: Dict[int, np.ndarray] = {}      # 只存用户输入的向量
        self.topics: List[Tuple[int, int]] = []            # [(start, end), ...]
        self.topics_summary: Dict[Tuple[int, int], str] = {}
        
        # 新增：连续低相似度计数和待定切换起始轮数       # NEW
        self._consecutive_low_count: int = 0               # NEW
        self._pending_topic_start: Optional[int] = None    # NEW
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 状态标记
        self._initialized = False
        self._last_processed_turn = 0

        self._load_summaries()

    def _load_summaries(self):
        """从idx文件加载已有的话题摘要
        
        格式：每行一个JSON，包含 start, end, summary
        """
        if not self.idx_file_path.exists():
            print(f"[TopicSegmenter] 摘要文件不存在，将创建新文件: {self.idx_file_path}")
            return
        
        try:
            with open(self.idx_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        start = data['start']
                        end = data['end']
                        summary = data['summary']
                        
                        # 存储摘要和话题范围
                        self.topics_summary[(start, end)] = summary
                        self.topics.append((start, end))
                        
                        # 更新最后处理的轮数
                        if end > self._last_processed_turn:
                            self._last_processed_turn = end
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"[TopicSegmenter] 第{line_num}行解析失败: {e}")
                        continue
            
            print(f"[TopicSegmenter] 已加载 {len(self.topics)} 个已有话题")
            
        except Exception as e:
            print(f"[TopicSegmenter] 加载idx文件失败: {e}")

    def _save_summary(self, start: int, end: int, summary: str):
        """保存话题摘要到idx文件（追加模式）
        
        Args:
            start: 话题起始轮数
            end: 话题结束轮数
            summary: 话题摘要
        """
        try:
            # 确保目录存在
            self.idx_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 追加写入
            with open(self.idx_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "start": start,
                    "end": end,
                    "summary": summary
                }, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"[TopicSegmenter] 保存话题摘要失败 [{start}-{end}]: {e}")

    def add_turn_vector(self, turn: int, vector: np.ndarray):
        """添加一轮对话的用户输入向量，用于话题检测
        
        Args:
            turn: 对话轮数
            vector: 用户输入对应的向量
        """
        if vector is None:
            print(f"[TopicSegmenter] 警告: 第{turn}轮用户向量为空")
            return
            
        with self.lock:
            # 存储向量
            self.turn_vectors[turn] = vector
            
            # 检测话题边界
            self._detect_boundary(turn)
            
            # 更新最后处理轮数
            if turn > self._last_processed_turn:
                self._last_processed_turn = turn

    def _detect_boundary(self, turn: int):
        """
        检测话题边界，并确保每个 turn 都属于一个话题。
        """
        # 1. 如果 topics 为空，开始第一个话题
        if not self.topics:
            self.topics.append((turn, None))
            print(f"[TopicSegmenter] 开始第一个话题: 第{turn}轮")
            return
    
        # 2. 获取最后一个话题
        last_topic = self.topics[-1]
        last_start, last_end = last_topic
    
        # 3. 如果最后一个话题已结束，直接开始新话题
        if last_end is not None:
            self.topics.append((turn, None))
            print(f"[TopicSegmenter] 开始新话题: 第{turn}轮")
            return
    
        # 4. 最后一个话题未结束，需要进行连续性检测
        # 检查上一轮向量是否存在（确保连续）
        if turn - 1 not in self.turn_vectors:
            # 缺少上一轮向量，可能是程序重启或向量丢失
            print(f"[TopicSegmenter] 警告: 缺少第{turn-1}轮的向量，无法检测连续性")
            # 为了不中断话题分割，我们结束当前话题并开始新话题
            self.topics[-1] = (last_start, turn - 1)  # 结束旧话题
            self._process_topic(last_start, turn - 1)
            self.topics.append((turn, None))          # 开始新话题
            print(f"[TopicSegmenter] 因缺少向量，强制切换话题: 旧话题 {last_start}-{turn-1} 结束，新话题从第{turn}轮开始")
            self._reset_pending()
            return
    
        # 5. 有上一轮向量，计算相似度
        prev_vec = self.turn_vectors[turn - 1]
        curr_vec = self.turn_vectors[turn]
        sim = self._cosine_similarity(prev_vec, curr_vec)
        print(f"[TopicSegmenter] 第{turn-1}轮与第{turn}轮相似度: {sim:.4f} (阈值: {self.similarity_threshold})")
    
        # 6. 相似度低于阈值 → 可能为新话题
        if sim < self.similarity_threshold:
            self._consecutive_low_count += 1
            if self._pending_topic_start is None:
                self._pending_topic_start = turn
                print(f"[TopicSegmenter] 可能的新话题起始于第{turn}轮 (连续第1次低)")
            else:
                print(f"[TopicSegmenter] 连续第{self._consecutive_low_count}次低相似度")
    
            # 达到连续阈值 → 确认话题切换
            if self._consecutive_low_count >= self.consecutive_low_threshold:
                self._confirm_topic_switch(turn)
        else:
            # 相似度恢复正常，重置待定状态
            if self._pending_topic_start is not None:
                print(f"[TopicSegmenter] 相似度恢复正常，取消待定的话题切换")
            self._reset_pending()

    def _confirm_topic_switch(self, current_turn: int):
        """确认话题切换，结束旧话题并开始新话题
        
        Args:
            current_turn: 当前轮数（达到连续阈值的最后一轮）
        """
        # 旧话题应为最后一个且未结束
        last_topic = self.topics[-1]
        if last_topic[1] is not None:
            # 理论上不应发生，但做防御
            print(f"[TopicSegmenter] 错误: 最后一个话题已结束，无法切换")
            self._reset_pending()
            return
        
        start_old, _ = last_topic
        # 新话题开始于第一次低相似度的轮数
        new_start = self._pending_topic_start
        end_old = new_start - 1
        
        # 结束旧话题
        self.topics[-1] = (start_old, end_old)
        # 处理已完成的话题（生成摘要）
        self._process_topic(start_old, end_old)
        
        # 开始新话题
        self.topics.append((new_start, None))
        
        print(f"[TopicSegmenter] 检测到话题切换: 旧话题 {start_old}-{end_old} 结束，新话题从第{new_start}轮开始")
        
        # 重置待定状态
        self._reset_pending()

    def _reset_pending(self):
        """重置连续低计数和待定起始点"""
        self._consecutive_low_count = 0
        self._pending_topic_start = None

    def _process_topic(self, start: int, end: int):
        """处理一个已完成的话题
        
        主要任务：
        1. 检查是否已处理过
        2. 获取对话内容
        3. 调用小模型生成摘要
        4. 保存摘要
        
        Args:
            start: 话题起始轮数
            end: 话题结束轮数
        """
        # 去重检查
        if (start, end) in self.topics_summary:
            print(f"[TopicSegmenter] 话题 {start}-{end} 已处理，跳过")
            return
        
        # 计算话题长度
        topic_length = end - start + 1
        
        # 只有足够长的话题才生成摘要（避免为短对话生成无意义摘要）
        if topic_length < 5:
            print(f"[TopicSegmenter] 话题 {start}-{end} 长度({topic_length})不足5轮，跳过摘要生成")
            return
            
        # 检查是否有小模型可用
        if self.small_llm_func is None:
            print(f"[TopicSegmenter] 未配置小模型，跳过话题 {start}-{end} 的摘要生成")
            return
        
        # 获取话题范围内的对话内容
        # 为了控制token使用，最多取前5轮对话作为样本
        dialogues = self.dialogue_manager.get_dialogues_in_range(start, min(start+4, end))
        
        if not dialogues:
            print(f"[TopicSegmenter] 无法获取话题 {start}-{end} 的对话内容")
            return
        
        # 构建提示
        dialog_texts = []
        for turn, user, ai in dialogues:
            dialog_texts.append(f"第{turn}轮\n用户：{user}\n助手：{ai}")
        
        prompt = (
            "请根据以下对话内容，用一句话概括这个话题的主要内容。\n"
            "要求：简洁、准确、不超过30个字。\n\n"
            f"{chr(10).join(dialog_texts)}\n\n"
            "概括："
        )
        
        # 调用小模型生成摘要
        try:
            summary = self.small_llm_func(prompt)
            
            if summary and len(summary.strip()) > 5:
                # 清理摘要
                summary = summary.strip().strip('"').strip("'")
                
                # 保存到内存和文件
                self.topics_summary[(start, end)] = summary
                self._save_summary(start, end, summary)
                
                # print(f"[TopicSegmenter] 话题 {start}-{end} 摘要生成成功: {summary}")
            else:
                print(f"[TopicSegmenter] 话题 {start}-{end} 生成的摘要无效")
                
        except Exception as e:
            print(f"[TopicSegmenter] 生成话题 {start}-{end} 摘要时出错: {e}")

    def finalize_topics(self):
        """结束所有未完成的话题（程序退出时调用）
        
        确保最后一个话题被正确处理并生成摘要
        """
        with self.lock:
            if not self.topics:
                print("[TopicSegmenter] 没有话题需要结束")
                return
            
            # 如果有待定切换，先确认它（强制结束旧话题并开始新话题？）
            # 简单起见，在finalize时直接结束最后一个话题，忽略待定状态
            if self._pending_topic_start is not None:
                print("[TopicSegmenter] 程序结束时存在待定话题切换，将直接结束当前话题")
                self._reset_pending()
            
            # 检查最后一个话题是否已结束
            last_topic = self.topics[-1]
            if last_topic[1] is None:
                start, _ = last_topic
                
                # 确定结束轮数
                if self.turn_vectors:
                    end = max(self.turn_vectors.keys())
                else:
                    end = start
                
                # 结束话题
                self.topics[-1] = (start, end)
                
                # 生成摘要
                self._process_topic(start, end)
                
                print(f"[TopicSegmenter] 已结束最后一个话题 {start}-{end}")
            
            # 统计
            completed = len([t for t in self.topics if t[1] is not None])
            print(f"[TopicSegmenter] 话题处理完成: 总计 {len(self.topics)} 个话题，{completed} 个已完结")

    def get_topic_for_turn(self, turn: int) -> Optional[Tuple[int, int]]:
        """获取指定轮数所属的话题范围
        
        Args:
            turn: 对话轮数
            
        Returns:
            (start, end) 或 None（如果未找到）
        """
        with self.lock:
            for start, end in self.topics:
                if end is None:
                    # 跳过未结束的话题
                    continue
                if start <= turn <= end:
                    return (start, end)
            return None

    def get_topics_in_range(self, start_turn: int, end_turn: int) -> List[Tuple[int, int]]:
        """获取与指定轮数范围有重叠的所有话题
        
        Args:
            start_turn: 起始轮数
            end_turn: 结束轮数
            
        Returns:
            话题范围列表，已裁剪到给定范围内
        """
        with self.lock:
            result = []
            for start, end in self.topics:
                if end is None:
                    # 跳过未结束的话题
                    continue
                    
                # 检查是否有重叠
                if not (end < start_turn or start > end_turn):
                    # 裁剪到给定范围内
                    overlap_start = max(start, start_turn)
                    overlap_end = min(end, end_turn)
                    result.append((overlap_start, overlap_end))
            
            # 按起始轮数排序
            result.sort(key=lambda x: x[0])
            return result

    def get_summary_for_topic(self, start: int, end: int) -> Optional[str]:
        """获取指定话题范围的摘要
        
        Args:
            start: 话题起始轮数
            end: 话题结束轮数
            
        Returns:
            摘要文本，如果没有则返回 None
        """
        return self.topics_summary.get((start, end))

    def get_all_topics(self) -> List[Tuple[int, int, Optional[str]]]:
        """获取所有已识别的话题及其摘要
        
        Returns:
            [(start, end, summary), ...] 列表
        """
        with self.lock:
            result = []
            for start, end in self.topics:
                if end is None:
                    # 未结束的话题
                    result.append((start, None, None))
                else:
                    summary = self.topics_summary.get((start, end))
                    result.append((start, end, summary))
            return result

    def get_stats(self) -> Dict[str, any]:
        """获取话题分割器的统计信息"""
        with self.lock:
            completed = [t for t in self.topics if t[1] is not None]
            return {
                "total_topics": len(self.topics),
                "completed_topics": len(completed),
                "topics_with_summary": len(self.topics_summary),
                "last_processed_turn": self._last_processed_turn,
                "similarity_threshold": self.similarity_threshold,
                "consecutive_low_threshold": self.consecutive_low_threshold,   # NEW
                "consecutive_low_count": self._consecutive_low_count,           # NEW
                "idx_file": str(self.idx_file_path),
                "has_small_llm": self.small_llm_func is not None
            }

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            v1: 向量1
            v2: 向量2
            
        Returns:
            相似度值（0-1之间）
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def clear(self):
        """清空所有数据（主要用于测试）"""
        with self.lock:
            self.turn_vectors.clear()
            self.topics.clear()
            self.topics_summary.clear()
            self._last_processed_turn = 0
            self._consecutive_low_count = 0          # NEW
            self._pending_topic_start = None         # NEW
            print("[TopicSegmenter] 已清空所有数据")