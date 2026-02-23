#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工具调用模块 - 处理Mori助手的记忆检索工具调用
"""

import json
import re
from typing import Optional, Dict, Any, List, Tuple

# ==================== 工具调用常量 ====================
TOOL_CALL_PATTERN = r'{"action":\s*"retrieve_memory",\s*"query":\s*"(.*?)"\s*}'
MAX_RETRIEVALS = 3

# ==================== 工具调用检测函数 ====================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取JSON格式的工具调用
    
    Args:
        text: 可能包含JSON的文本
        
    Returns:
        解析后的JSON字典，如果未找到则返回None
    """
    # 尝试匹配完整的JSON对象
    stack = []
    start = None
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    try:
                        json_str = text[start:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
    
    # 使用正则表达式作为备选方案
    try:
        match = re.search(r'\{.*?"action".*?:.*?"retrieve_memory".*?,.*?"query".*?:.*?"(.*?)".*?\}', text, re.DOTALL)
        if match:
            query_content = match.group(1)
            query_content = query_content.replace('"', '\\"')
            fixed_json = f'{{"action": "retrieve_memory", "query": "{query_content}"}}'
            return json.loads(fixed_json)
    except Exception:
        pass
    
    return None

def detect_tool_call(text: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    检测文本中是否包含工具调用
    
    Args:
        text: 要检测的文本
        
    Returns:
        (是否检测到工具调用, 查询内容, 完整的工具调用字典)
    """
    tool_call = extract_json_from_text(text)
    if tool_call and tool_call.get("action") == "retrieve_memory":
        query = tool_call.get("query", "").strip()
        return True, query, tool_call
    return False, None, None

def should_force_retrieval(user_input: str, trigger_words: List[str]) -> bool:
    """
    判断是否应该强制触发检索
    
    Args:
        user_input: 用户输入
        trigger_words: 触发词列表
        
    Returns:
        是否应该强制检索
    """
    return any(word in user_input for word in trigger_words)

# ==================== 工具调用响应构建 ====================

def build_memory_retrieval_response(
    memory_module: Any,
    query: str,
    max_results: int = 6
) -> str:
    """
    构建记忆检索响应
    
    Args:
        memory_module: 记忆模块实例
        query: 搜索查询
        max_results: 最大结果数
        
    Returns:
        格式化的检索结果响应
    """
    print(f"[工具调用] 开始检索查询: {query}")
    
    try:
        # 使用记忆模块搜索
        results = memory_module.search_original_memories(query_text=query, max_results=max_results)
        
        if not results:
            print(f"[工具调用] 未找到相关记忆")
            return "【检索结果】\n没有找到相关的记忆。你可以继续推理。"
        
        print(f"[工具调用] 找到 {len(results)} 条相关记忆")
        
        # 格式化结果
        memories_text = "\n".join([
            f"{i+1}. 【轮数: {mem.created_turn}】【相似度: {score:.3f}】\n"
            f"   事实: {mem.user_input[:100]}{'...' if len(mem.user_input) > 100 else ''}"
            for i, (mem, score) in enumerate(results)
        ])
        
        return f"""【检索结果】
检索到以下相关原子事实（按相关性排序）:
{memories_text}

现在你可以结合这些记忆继续逐步推理，并给出最终回答。"""
    
    except Exception as e:
        print(f"[工具调用] 检索过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return "【检索结果】\n记忆检索过程中出现错误。请继续推理。"

def build_force_retrieval_message(query: str) -> str:
    """
    构建强制检索的助手消息
    
    Args:
        query: 查询内容
        
    Returns:
        格式化的助手消息
    """
    return f'{{"action": "retrieve_memory", "query": "{query}"}}'

def build_system_tool_response(
    query: str,
    retrieval_count: int,
    max_retrievals: int,
    memory_module: Any
) -> str:
    """
    构建系统工具响应
    
    Args:
        query: 查询内容
        retrieval_count: 当前检索次数
        max_retrievals: 最大检索次数
        memory_module: 记忆模块实例
        
    Returns:
        系统工具响应
    """
    if retrieval_count > max_retrievals:
        return "\n【系统提示】已达到最大检索次数，请直接基于现有信息回答。\n"
    else:
        return "\n" + build_memory_retrieval_response(memory_module, query) + "\n"

# ==================== 消息处理函数 ====================

def prepare_messages_for_tool_call(
    messages: List[Dict],
    full_response: str,
    tool_query: str,
    forced_mode: bool
) -> List[Dict]:
    """
    准备进行工具调用后的消息列表
    
    Args:
        messages: 原始消息列表
        full_response: 已生成的响应
        tool_query: 工具查询内容
        forced_mode: 是否为强制检索模式
        
    Returns:
        更新后的消息列表
    """
    new_messages = messages.copy()
    
    if forced_mode and new_messages and new_messages[-1]["role"] == "assistant" and \
       new_messages[-1]["content"].endswith('{"action": "retrieve_memory", "query": "'):
        new_messages[-1]["content"] = f'{{"action": "retrieve_memory", "query": "{tool_query}"}}'
    else:
        new_messages.append({"role": "assistant", "content": full_response})
    
    return new_messages

# ==================== 工具调用处理主函数 ====================

def handle_tool_call(
    llm: Any,
    messages: List[Dict],
    full_response: str,
    tool_query: str,
    retrieval_count: int,
    max_retrievals: int,
    memory_module: Any,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    frequency_penalty: float = 0.3,
    max_new_tokens: int = 512
) -> Tuple[str, int]:
    """
    处理工具调用并生成最终响应
    
    Args:
        llm: 语言模型实例
        messages: 消息列表
        full_response: 已生成的响应
        tool_query: 工具查询内容
        retrieval_count: 当前检索次数
        max_retrievals: 最大检索次数
        memory_module: 记忆模块实例
        temperature: 温度参数
        top_p: top_p参数
        repeat_penalty: 重复惩罚
        frequency_penalty: 频率惩罚
        max_new_tokens: 最大生成token数
        
    Returns:
        (最终响应, 更新后的检索次数)
    """
    retrieval_count += 1
    
    print(f"\n[工具调用] 捕获工具调用，正在检索... (次数: {retrieval_count}/{max_retrievals})")
    
    # 构建系统响应
    tool_response = build_system_tool_response(
        query=tool_query,
        retrieval_count=retrieval_count,
        max_retrievals=max_retrievals,
        memory_module=memory_module
    )
    
    print(tool_response, end="", flush=True)
    
    # 准备新消息
    new_messages = prepare_messages_for_tool_call(
        messages=messages,
        full_response=full_response,
        tool_query=tool_query,
        forced_mode=False  # 这里可以根据实际情况传入
    )
    new_messages.append({"role": "system", "content": tool_response})
    
    # 生成最终响应
    final_stream = llm.create_chat_completion(
        messages=new_messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        frequency_penalty=frequency_penalty,
        stream=True,
        top_k=0,
        min_p=0.05,
        seed=-1,
    )
    
    final_answer = ""
    for chunk in final_stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            print(delta, end="", flush=True)
            final_answer += delta
    
    return final_answer, retrieval_count

# ==================== 工具调用配置类 ====================

class ToolConfig:
    """工具调用配置类"""
    
    def __init__(
        self,
        trigger_words: List[str],
        max_retrievals: int = 3,
        enabled: bool = True
    ):
        """
        初始化工具配置
        
        Args:
            trigger_words: 触发词列表
            max_retrievals: 最大检索次数
            enabled: 是否启用工具调用
        """
        self.trigger_words = trigger_words
        self.max_retrievals = max_retrievals
        self.enabled = enabled
    
    @classmethod
    def from_list(cls, trigger_words: List[str], max_retrievals: int = 3) -> 'ToolConfig':
        """从触发词列表创建配置"""
        return cls(trigger_words=trigger_words, max_retrievals=max_retrievals)

# ==================== 工具调用管理器 ====================

class ToolCallManager:
    """工具调用管理器"""
    
    def __init__(self, config: ToolConfig):
        """
        初始化工具调用管理器
        
        Args:
            config: 工具调用配置
        """
        self.config = config
        self.retrieval_count = 0
    
    def reset(self):
        """重置检索计数"""
        self.retrieval_count = 0
    
    def should_force_retrieval(self, user_input: str) -> bool:
        """判断是否需要强制检索"""
        return any(word in user_input for word in self.config.trigger_words)
    
    def detect_in_stream(self, full_response: str, forced_mode: bool = False) -> Tuple[bool, Optional[str]]:
        """
        在流式响应中检测工具调用
        
        Args:
            full_response: 累积的响应文本
            forced_mode: 是否为强制检索模式
            
        Returns:
            (是否检测到工具调用, 查询内容)
        """
        if not self.config.enabled:
            return False, None
        
        if forced_mode:
            text_to_check = '{"action": "retrieve_memory", "query": "' + full_response
        else:
            text_to_check = full_response
        
        detected, query, _ = detect_tool_call(text_to_check)
        return detected, query
    
    def can_retry(self) -> bool:
        """检查是否还可以重试"""
        return self.retrieval_count < self.config.max_retrievals
    
    def increment_count(self):
        """增加检索计数"""
        self.retrieval_count += 1
    
    def get_count(self) -> int:
        """获取当前检索次数"""
        return self.retrieval_count
    
    def get_remaining(self) -> int:
        """获取剩余可检索次数"""
        return max(0, self.config.max_retrievals - self.retrieval_count)


# 导出主要函数和类
__all__ = [
    'ToolConfig',
    'ToolCallManager',
    'extract_json_from_text',
    'detect_tool_call',
    'should_force_retrieval',
    'build_memory_retrieval_response',
    'build_force_retrieval_message',
    'build_system_tool_response',
    'prepare_messages_for_tool_call',
    'handle_tool_call',
    'MAX_RETRIEVALS'
]