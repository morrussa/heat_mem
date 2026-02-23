# memory_system/infrastructure/dialogue_manager.py
from pathlib import Path
from typing import Optional, List, Tuple


class DialogueManager:
    """只管理原始对话的存储和读取，不涉及任何记忆ID"""
    
    # 使用 ASCII 控制字符作为分隔符
    FIELD_SEP = '\x1F'  # 单元分隔符
    RECORD_SEP = '\n'  # 记录分隔符
    
    def __init__(self, history_file: str = "./memory/history.txt"):
        self.history_file = Path(history_file)
        self._init_file()
    
    def _init_file(self):
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self.history_file.write_text("")
    
    def add_dialogue(self, turn: int, user_input: str, ai_response: str):
        """追加原始对话记录，使用分隔符"""
        try:
            # 直接使用分隔符连接，不转义
            line = f"{turn}{self.FIELD_SEP}{user_input}{self.FIELD_SEP}{ai_response}{self.RECORD_SEP}"
            
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(line)
                
        except Exception as e:
            print(f"[DialogueManager] 写入失败: {e}")
    
    def get_dialogue(self, turn: int) -> Optional[Tuple[str, str]]:
        """根据轮数获取原始对话 (user_input, ai_response)"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按记录分隔符分割
            records = content.split(self.RECORD_SEP)
            
            for record in records:
                if not record:
                    continue
                
                # 按字段分隔符分割
                parts = record.split(self.FIELD_SEP)
                if len(parts) >= 3:
                    try:
                        file_turn = int(parts[0])
                        if file_turn == turn:
                            return (parts[1], parts[2])
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"[DialogueManager] 读取失败: {e}")
        return None
    
    def get_dialogues_in_range(self, start_turn: int, end_turn: int) -> List[Tuple[int, str, str]]:
        """获取指定轮数范围内的所有原始对话"""
        dialogues = []
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            records = content.split(self.RECORD_SEP)
            
            for record in records:
                if not record:
                    continue
                
                parts = record.split(self.FIELD_SEP)
                if len(parts) >= 3:
                    try:
                        turn = int(parts[0])
                        if start_turn <= turn <= end_turn:
                            dialogues.append((turn, parts[1], parts[2]))
                    except (ValueError, IndexError):
                        continue
            
            dialogues.sort(key=lambda x: x[0])
            
        except Exception as e:
            print(f"[DialogueManager] 批量读取失败: {e}")
        
        return dialogues
    
    def search_by_keyword(self, keyword: str, max_results: int = 50) -> List[Tuple[int, str, str]]:
        """按关键词搜索原始对话内容"""
        results = []
        keyword_lower = keyword.lower()
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            records = content.split(self.RECORD_SEP)
            
            # 从最新的开始搜索
            for record in reversed(records):
                if not record:
                    continue
                
                parts = record.split(self.FIELD_SEP)
                if len(parts) >= 3:
                    try:
                        turn = int(parts[0])
                        user_input = parts[1]
                        ai_response = parts[2]
                        
                        if keyword_lower in user_input.lower() or keyword_lower in ai_response.lower():
                            results.append((turn, user_input, ai_response))
                            if len(results) >= max_results:
                                break
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"[DialogueManager] 搜索失败: {e}")
        
        return results
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        try:
            if not self.history_file.exists():
                return {"total_lines": 0, "first_turn": None, "last_turn": None}
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            records = [r for r in content.split(self.RECORD_SEP) if r]
            
            if not records:
                return {"total_lines": 0, "first_turn": None, "last_turn": None}
            
            first_parts = records[0].split(self.FIELD_SEP)
            last_parts = records[-1].split(self.FIELD_SEP)
            
            first_turn = int(first_parts[0]) if len(first_parts) >= 3 else None
            last_turn = int(last_parts[0]) if len(last_parts) >= 3 else None
            
            return {
                "total_lines": len(records),
                "first_turn": first_turn,
                "last_turn": last_turn
            }
            
        except Exception as e:
            print(f"[DialogueManager] 获取统计失败: {e}")
            return {}