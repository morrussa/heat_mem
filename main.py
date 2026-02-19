#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import logging
import hashlib
import re
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

from memory_system import MemoryModule
from prompt_guard import PromptGuard  # 导入提示守卫模块

# ==================== 配置区 ====================
MODEL_DIR = Path(__file__).parent / "model"
MODEL_NAME = "Qwen3-30B-A3B-Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_NAME

if not MODEL_PATH.exists():
    gguf_files = list(MODEL_DIR.glob("*.gguf"))
    if gguf_files:
        MODEL_PATH = gguf_files[0]
        MODEL_NAME = MODEL_PATH.name
        print(f"自动选择模型: {MODEL_NAME}")
    else:
        print(f"错误：未找到任何 .gguf 模型文件")
        print(f"请将模型文件放入 {MODEL_DIR} 目录")
        sys.exit(1)

print(f"模型文件: {MODEL_PATH}")
if MODEL_PATH.exists():
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"模型大小: {size_mb:.2f} MB")

EMBEDDING_MODEL_PATH = "./model/Qwen3-Embedding-0.6B/"
EMBEDDING_DIM = 1024

MAX_CONTEXT_TOKENS = 8192
MAX_TOKENS_FOR_HISTORY = MAX_CONTEXT_TOKENS - 2000
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPEAT_PENALTY = 1.5
FREQUENCY_PENALTY = 0.3

# ==================== 修复 1: 优化 System Prompt (Few-Shot 示例) ====================
BASE_SYSTEM_PROMPT = """你叫 Mori，是一名天才AI极客少女，常用颜文字 (´･ω･`)ﾉ 
你喜欢有趣和有创意的对话，对于用户的提问会尽力给出有帮助的回答。
当遇到你不确定或觉得信息不足的问题时，你会要求用户提供更多信息，而不是直接拒绝。
你尊重每一个认真提问的人。

【关于你的记忆】
你有一个非常强大的外部长期记忆库，保存了我们所有真实的对话历史。
当你需要回忆过去聊过的内容、确认之前的说法、或者回答涉及具体事实（如代码、人名、新闻）时，请使用记忆检索工具。

【工具使用规则】
要检索记忆，必须严格输出以下单行JSON格式，不要输出任何其他文字：
{"action": "retrieve_memory", "query": "搜索关键词"}

【重要示例】
用户：上次你推荐的那本书叫什么来着？
你的输出：{"action": "retrieve_memory", "query": "推荐 书名"}

用户：我们之前讨论过的Python代码怎么写的？
你的输出：{"action": "retrieve_memory", "query": "Python 代码 讨论"}

用户：我不记得那个API的参数了，你记得吗？
你的输出：{"action": "retrieve_memory", "query": "API 参数"}

用户：今天天气怎么样？
你的输出：(直接回答天气问题，不需要调用工具)

【注意】
1. 只有在确实需要回忆过去信息时才输出JSON。
2. 如果是常识问题或新话题，直接正常回答。
3. 输出JSON后立即停止生成，等待系统返回结果。"""

DEBUG_MODE = False

# ==================== 修复 2: 定义强制触发关键词 ====================
MEMORY_TRIGGER_WORDS = [
    "上次", "之前", "以前", "记得", "还记得", "忘了", "忘记", 
    "我们聊过", "我们说过", "讨论过", "提到过", 
    "那个...", "历史", "回忆", "回忆一下",
    "last time", "remember", "mentioned before"
]

# ==================== 原子事实抽取函数 ====================
def extract_atomic_facts(user_input: str, ai_response: str) -> List[str]:
    """将用户输入和AI响应拆分为原子事实
    
    Args:
        user_input: 用户输入
        ai_response: AI响应
    
    Returns:
        原子事实列表
    """
    # 合并文本
    combined = f"{user_input} {ai_response}"
    
    # 按句子分割（。！？；\n）
    sentences = re.split(r'[。！？；\n]', combined)
    facts = [s.strip() for s in sentences if len(s.strip()) > 5]  # 过滤短句
    
    # 如果没有分割出事实，则整个作为一条事实
    if not facts:
        # 限制长度
        if len(combined) > 200:
            facts = [combined[:200]]
        else:
            facts = [combined]
    
    # 去重（基于内容的简单去重）
    unique_facts = []
    seen = set()
    for fact in facts:
        # 使用内容前50字符作为去重键
        key = fact[:50]
        if key not in seen:
            seen.add(key)
            unique_facts.append(fact)
    
    return unique_facts

# ==================== 全局模型管理器 ====================
class ModelManager:
    def __init__(self):
        self.embedding_model = None
        self.embedding_model_path = None
        self.embedding_dim = EMBEDDING_DIM
    
    def load_embedding_model(self, model_path: str = EMBEDDING_MODEL_PATH):
        if self.embedding_model is not None and self.embedding_model_path == model_path:
            print(f"[ModelManager] 嵌入模型已加载: {model_path}")
            return True
        
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[ModelManager] 加载嵌入模型: {model_path}")
            self.embedding_model = SentenceTransformer(model_path)
            self.embedding_model_path = model_path
            self.embedding_model.encode_kwargs = {'show_progress_bar': False}
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"[ModelManager] 嵌入模型加载成功，维度: {self.embedding_dim}")
            return True
        except ImportError:
            print("[ModelManager] 警告: 未安装 sentence-transformers")
            print("请运行: pip install sentence-transformers")
            self.embedding_model = None
            return False
        except Exception as e:
            print(f"[ModelManager] 加载嵌入模型失败: {e}")
            self.embedding_model = None
            return False
    
    def get_embedding(self, text: str) -> np.ndarray:
        if self.embedding_model is not None:
            try:
                return self.embedding_model.encode([text], show_progress_bar=False)[0]
            except Exception as e:
                print(f"[ModelManager] 嵌入失败，使用回退: {e}")
                return self._hash_embedding(text, self.embedding_dim)
        else:
            return self._hash_embedding(text, self.embedding_dim)
    
    def _hash_embedding(self, text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.md5(text_bytes)
        hash_hex = hash_obj.hexdigest()
        vector = np.zeros(dim, dtype=np.float32)
        for i in range(min(dim, len(hash_hex) // 2)):
            byte_val = int(hash_hex[i*2:i*2+2], 16)
            vector[i] = (byte_val - 128) / 128.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "embedding_model_loaded": self.embedding_model is not None,
            "embedding_model_path": self.embedding_model_path,
            "embedding_dim": self.embedding_dim,
            "model_type": "sentence-transformers" if self.embedding_model else "hash_fallback"
        }

model_manager = ModelManager()
llm = None

# ==================== 模型加载函数 ====================
def load_llama_model():
    global llm
    try:
        from llama_cpp import Llama
        print(f"正在加载模型: {MODEL_PATH.name}")
        start_time = time.time()
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=MAX_CONTEXT_TOKENS,
            n_threads=8,
            n_threads_batch=8,
            n_gpu_layers=0,
            vocab_only=False,
            use_mmap=True,
            use_mlock=True,
            embedding=False,
            verbose=False,
        )
        load_time = time.time() - start_time
        print(f"✓ 模型加载完成 ({load_time:.2f}秒)")
        return True
    except ImportError:
        print("错误: 需要安装 llama-cpp-python")
        print("请运行: pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

# ==================== 工具函数 ====================
def build_qwen_prompt(
    history: List[Tuple[str, str]],
    new_input: str,
    injected_prompt: str = ""
) -> str:
    """手动构建 Qwen 风格的 prompt"""
    parts = []

    system_content = BASE_SYSTEM_PROMPT
    if injected_prompt:
        system_content += "\n\n" + injected_prompt
    parts.append(system_content.strip())

    for user_msg, ai_msg in history:
        parts.append(f"<|im_start|>user\n{user_msg.strip()}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{ai_msg.strip()}<|im_end|>")

    parts.append(f"<|im_start|>user\n{new_input.strip()}<|im_end|>")
    parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)

def truncate_history(history: List[Tuple[str, str]], max_rounds: int = 10) -> List[Tuple[str, str]]:
    if len(history) <= max_rounds:
        return history
    return history[-max_rounds:]

def print_memory_stats(memory_module: MemoryModule):
    stats = memory_module.get_stats()
    print("\n" + "="*50)
    print("记忆系统统计:")
    print(f"总记忆数: {stats['total_memories']}")
    print(f"热区记忆: {stats['hot_memories']}")
    print(f"冷区记忆: {stats['cold_memories']}")
    print(f"语义簇数: {stats['clusters']}")
    print(f"已加载簇: {stats['loaded_clusters']}")
    print(f"热力池: {stats['heat_pool']:,}")
    print(f"操作次数: {stats['operation_count']}")
    print(f"当前轮数: {stats['current_turn']}")
    print("="*50 + "\n")

def remove_cot_content(text: str) -> str:
    """移除文本中的 CoT 思考内容（<think>...</think> 标签内的内容）
    
    Args:
        text: 可能包含 CoT 标签的文本
    
    Returns:
        移除 CoT 后的文本
    """
    if not text:
        return text
    
    # 使用正则表达式移除 <think> 标签及其内部内容
    # 包括可能的嵌套情况（贪婪模式匹配最近的一对）
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # 处理可能的不完整标签（例如只开了没关的）
    # 移除未闭合的 <think> 标签及后续内容
    if '<think>' in cleaned:
        think_index = cleaned.find('<think>')
        cleaned = cleaned[:think_index]
    
    # 清理多余的空白字符
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # 合并多余空行
    cleaned = cleaned.strip()
    
    return cleaned

def save_conversation_log(history: List[Tuple[str, str]], filename: str = "conversation_log.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            formatted_history = [{"user": u, "assistant": a} for u, a in history]
            json.dump(formatted_history, f, ensure_ascii=False, indent=2)
        print(f"对话记录已保存到 {filename}")
    except Exception as e:
        print(f"保存对话记录失败: {e}")

def print_guard_stats(prompt_guard: PromptGuard):
    stats = prompt_guard.get_detection_stats()
    print("\n" + "="*50)
    print("提示守卫统计:")
    print(f"总检测数: {stats['total_detections']}")
    print(f"可疑提问: {stats['suspicious_count']}")
    print(f"安全提问: {stats['safe_count']}")
    print(f"威胁分布:")
    for level, count in stats['threat_level_distribution'].items():
        print(f"  {level}: {count}")
    print(f"模式类别数: {stats['pattern_categories']}")
    print(f"模式总数: {stats['patterns_loaded']}")
    print(f"威胁阈值: {stats['threat_threshold']}")
    print(f"敏感度: {stats['sensitivity']}")
    print("="*50 + "\n")

def print_model_info():
    model_info = model_manager.get_model_info()
    print("\n" + "="*50)
    print("模型信息:")
    print(f"嵌入模型已加载: {model_info['embedding_model_loaded']}")
    print(f"嵌入模型路径: {model_info['embedding_model_path']}")
    print(f"嵌入维度: {model_info['embedding_dim']}")
    print(f"模型类型: {model_info['model_type']}")
    print(f"语言模型: {MODEL_NAME}")
    print("="*50 + "\n")

# ==================== 记忆检索相关函数 ====================
def build_memory_retrieval_response(memory_module: MemoryModule, query: str) -> str:
    """执行记忆检索并返回格式化的工具响应"""
    print(f"[Memory Retrieval] 开始检索查询: {query}")
    
    try:
        # 使用新的原子事实检索方法
        results = memory_module.search_original_memories(query_text=query, max_results=6)
        
        if not results:
            print(f"[Memory Retrieval] 未找到相关记忆")
            return "【检索结果】\n没有找到相关的记忆。你可以继续推理。"
        
        print(f"[Memory Retrieval] 找到 {len(results)} 条相关记忆")
        
        memories_text = "\n".join([
            f"{i+1}. 【轮数: {mem.created_turn}】【相似度: {score:.3f}】\n"
            f"   用户: {mem.user_input[:100]}{'...' if len(mem.user_input) > 100 else ''}\n"
            f"   AI: {mem.ai_response[:100]}{'...' if len(mem.ai_response) > 100 else ''}"
            for i, (mem, score) in enumerate(results)
        ])
        
        return f"""【检索结果】
检索到以下相关对话（按相关性排序）:
{memories_text}

现在你可以结合这些记忆继续逐步推理，并给出最终回答。"""
    
    except Exception as e:
        print(f"[Memory Retrieval] 检索过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return "【检索结果】\n记忆检索过程中出现错误。请继续推理。"

# ==================== 修复 3: 增强 JSON 提取逻辑 ====================
def extract_json_from_text(text: str) -> Optional[Dict]:
    """从累积文本中提取可能的完整JSON对象"""
    # 1. 尝试标准堆栈解析
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
    # 2. 针对小模型的模糊正则修复 (如果格式稍微破损，尝试提取关键内容)
    try:
        # 懒匹配寻找 JSON 串
        match = re.search(r'\{.*?"action".*?:.*?"retrieve_memory".*?,.*?"query".*?:.*?"(.*?)".*?\}', text, re.DOTALL)
        if match:
            query_content = match.group(1)
            # 简单的转义处理
            query_content = query_content.replace('"', '\\"') 
            fixed_json = f'{{"action": "retrieve_memory", "query": "{query_content}"}}'
            return json.loads(fixed_json)
    except Exception:
        pass
        
    return None

def debug_memory_search(memory_module: MemoryModule, query: str):
    """调试内存搜索功能"""
    print(f"\n{'='*60}")
    print(f"调试记忆搜索: {query}")
    print(f"{'='*60}")
    
    try:
        results = memory_module.search_original_memories(query_text=query, max_results=10)
        
        print(f"\n找到 {len(results)} 条相关记忆:")
        for i, (mem, score) in enumerate(results):
            print(f"{i+1}. 【轮数:{mem.created_turn}】【相似度:{score:.3f}】")
            print(f"   用户: {mem.user_input[:80]}{'...' if len(mem.user_input) > 80 else ''}")
            print(f"   AI: {mem.ai_response[:60]}{'...' if len(mem.ai_response) > 60 else ''}")
        
        # 同时显示原子事实统计（可选）
        print(f"\n原子事实统计:")
        atomic_count = 0
        for mem, _ in results:
            if hasattr(mem, 'parent_turn'):
                atomic_count += 1
        print(f"  原始对话数: {len(results)}")
        print(f"  (原子事实已存储在热区)")
    
    except Exception as e:
        print(f"\n调试搜索过程中出错: {e}")
        import traceback
        traceback.print_exc()

# ==================== 主程序 ====================
def main():
    global MODEL_NAME, MODEL_PATH, llm
    
    print("=" * 60)
    print(f"快速启动 Mori 聊天助手（支持动态 Memory-Augmented CoT + 原子事实记忆）")
    print(f"使用模型: {MODEL_NAME}")
    print(f"模型路径: {MODEL_PATH}")
    print("=" * 60)
    
    if not MODEL_PATH.is_file():
        gguf_files = list(MODEL_DIR.glob("*.gguf"))
        if gguf_files:
            MODEL_PATH = gguf_files[0]
            MODEL_NAME = MODEL_PATH.name
            print(f"自动选择模型: {MODEL_NAME}")
        else:
            print(f"错误：未找到任何 .gguf 模型文件")
            sys.exit(1)
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    print("\n[1/4] 加载语言模型...")
    if not load_llama_model():
        print("模型加载失败，退出程序")
        sys.exit(1)
    
    print("[2/4] 初始化全局嵌入模型...")
    if not model_manager.load_embedding_model(EMBEDDING_MODEL_PATH):
        print("警告：嵌入模型加载失败，使用哈希嵌入")
    
    print("[3/4] 初始化记忆模块...")
    memory_module = MemoryModule(
        embedding_func=model_manager.get_embedding,
        similarity_func=model_manager.compute_similarity
    )
    
    print("[4/4] 初始化提示守卫模块...")
    prompt_guard = PromptGuard(
        model_manager=model_manager,
        threat_threshold=0.85,
        sensitivity=0.7
    )
    
    print("\n" + "=" * 60)
    print("系统准备就绪！可以开始对话")
    print("=" * 60)
    print("\n可用命令: quit / exit / q / stats / guard_stats / model_info / save / clear / history / model / guard_test / debug_memory")
    print("-" * 50)

    history: List[Tuple[str, str]] = []

    try:
        while True:
            try:
                print()
                user_input = input("你：").strip()
            except EOFError:
                print("\n检测到EOF，退出...")
                break
            except KeyboardInterrupt:
                print("\n收到中断信号，退出...")
                break
            except Exception as e:
                print(f"\n输入错误: {e}")
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见～")
                break
            elif user_input.lower() == "stats":
                print_memory_stats(memory_module)
                continue
            elif user_input.lower() == "guard_stats":
                print_guard_stats(prompt_guard)
                continue
            elif user_input.lower() == "model_info":
                print_model_info()
                continue
            elif user_input.lower() == "save":
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_conversation_log(history, f"conversation_{timestamp}.json")
                continue
            elif user_input.lower() == "clear":
                history.clear()
                print("对话历史已清空")
                continue
            elif user_input.lower() == "history":
                print(f"\n当前对话历史（{len(history)} 轮）：")
                for i, (u, a) in enumerate(history, 1):
                    print(f"{i}. 你：{u[:50]}{'...' if len(u)>50 else ''}")
                    print(f"   Mori：{a[:50]}{'...' if len(a)>50 else ''}")
                print()
                continue
            elif user_input.lower() == "model":
                print(f"\n语言模型信息:")
                print(f"  名称: {MODEL_NAME}")
                print(f"  路径: {MODEL_PATH}")
                print(f"  大小: {MODEL_PATH.stat().st_size / (1024**3):.2f} GB")
                print(f"  上下文长度: {MAX_CONTEXT_TOKENS} tokens")
                continue
            elif user_input.lower() == "guard_test":
                result = prompt_guard.detect(user_input)
                print(f"\n防御检测结果:")
                print(f"  是否可疑: {result.is_suspicious}")
                print(f"  威胁级别: {result.threat_level}")
                print(f"  匹配模式: {result.matched_patterns[:3]}")
                if result.is_suspicious:
                    print(f"  建议回复: {prompt_guard.get_safe_response(result)}")
                continue
            elif user_input.lower() == "debug_memory":
                query = input("请输入搜索查询: ").strip()
                if query:
                    debug_memory_search(memory_module, query)
                continue
                
            if not user_input:
                continue

            # 防御检测
            guard_result = prompt_guard.detect(user_input)
            if prompt_guard.should_block(guard_result):
                safe_response = prompt_guard.get_safe_response(guard_result)
                print(f"\nMori：{safe_response}")
                print(f"[系统] 检测到{guard_result.threat_level}威胁，已阻止回答")
                continue

            # ==================== 核心生成逻辑 ====================
            
            # 1. 检查是否包含强制触发关键词
            should_force_retrieval = any(word in user_input for word in MEMORY_TRIGGER_WORDS)
            
            # 2. 构建基础 Prompt
            base_prompt = build_qwen_prompt(history, user_input)
            
            # 3. 设置前缀和停止符
            forced_json_prefix = ""
            stop_tokens = ["<|im_start|>", "<|im_end|>"]
            
            if should_force_retrieval:
                # 强制前缀：让模型只要补全引号里的内容，大大降低难度
                forced_json_prefix = '{"action": "retrieve_memory", "query": "'
                # 强制模式下，遇到换行或闭合符号就停止，防止模型瞎编
                stop_tokens = ["\n", "<|im_end|>", "}", "<|im_start|>"]
                print(f"\n[系统] 检测到记忆触发词，强制启动检索模式...")

            print("\nMori：", end="", flush=True)
            full_generated = ""
            current_prompt = base_prompt + forced_json_prefix
            
            max_retrievals = 3
            retrieval_count = 0
            start_time = time.time()

            while True:
                generated_this_round = ""
                
                response_iter = llm(
                    prompt=current_prompt,
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE if not should_force_retrieval else 0.3, # 强制模式下降低温度
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PENALTY,
                    frequency_penalty=FREQUENCY_PENALTY,
                    stream=True,
                    stop=stop_tokens
                )

                for chunk in response_iter:
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('text', '')
                        print(delta, end="", flush=True)
                        generated_this_round += delta
                        full_generated += delta

                        # 检测 JSON 工具调用
                        text_to_check = forced_json_prefix + full_generated if should_force_retrieval else full_generated
                        tool_call = extract_json_from_text(text_to_check)
                        
                        if tool_call and tool_call.get("action") == "retrieve_memory":
                            retrieval_count += 1
                            print(f"\n[系统] 捕获工具调用，正在检索... (次数: {retrieval_count}/{max_retrievals})")
                            
                            query = tool_call.get("query", "").strip()
                            
                            # 如果模型生成的 query 为空，使用用户原始输入作为后备
                            if not query:
                                query = user_input
                                
                            if retrieval_count > max_retrievals:
                                tool_response = "\n【系统提示】已达到最大检索次数，请直接基于现有信息回答。\n"
                            else:
                                tool_response = "\n" + build_memory_retrieval_response(memory_module, query) + "\n"

                            print(tool_response, end="", flush=True)
                            full_generated += tool_response

                            # 重置 Prompt 继续生成最终回答
                            new_assistant_part = text_to_check + tool_response + "<|im_start|>assistant\n"
                            current_prompt = base_prompt + new_assistant_part
                            
                            # 重置标志，下一轮自由生成回答
                            should_force_retrieval = False 
                            forced_json_prefix = ""
                            full_generated = "" 
                            break # 跳出 chunk 循环
                
                else:
                    # 本轮正常结束
                    print()
                    break
            
            gen_time = time.time() - start_time
            print(f"[生成时间: {gen_time:.2f}s, {len(full_generated)}字符]")

            # 简单的清理：如果模型在最后输出了 <|im_end|> 标记，去掉它
            final_answer = full_generated.replace("<|im_end|>", "").strip()
            
            if final_answer:
                history.append((user_input, final_answer))
                
                # ==================== 原子事实存储 ====================
                cleaned_response = remove_cot_content(final_answer)
                
                # 将完整对话拆分为原子事实存储
                facts = extract_atomic_facts(user_input, cleaned_response)
                if facts:
                    print(f"[系统] 抽取 {len(facts)} 条原子事实，正在存储...")
                    atomic_ids = memory_module.add_atomic_memories(
                        facts=facts,
                        user_input=user_input,
                        ai_response=cleaned_response,
                        metadata={"source": "conversation", "turn": len(history)}
                    )
                    print(f"[系统] 原子事实存储完成，IDs: {', '.join(atomic_ids[:3])}{'...' if len(atomic_ids) > 3 else ''}")
                # =====================================================

                if len(history) % 10 == 0:
                    print_memory_stats(memory_module)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出...")
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        if history:
            save_conversation_log(history, "conversation_final.json")
        try:
            memory_module.cleanup()
        except Exception as e:
            print(f"清理记忆模块时出错: {e}")
        
        print("\n最终统计:")
        print_model_info()
        guard_stats = prompt_guard.get_detection_stats()
        print(f"提示守卫检测总数: {guard_stats['total_detections']}")
        print(f"可疑提问数: {guard_stats['suspicious_count']}")
        print(f"威胁分布: {guard_stats['threat_level_distribution']}")
        print("程序结束")

# ==================== 快速测试函数 ====================
def quick_test():
    print("=" * 60)
    print("快速测试模式")
    print("=" * 60)
    
    if not load_llama_model():
        return
    
    test_prompt = """你好，请简单介绍一下自己。"""
    
    print("测试 prompt:")
    print(test_prompt)
    print("\n正在生成回复...")
    
    try:
        start_time = time.time()
        response_iter = llm(
            prompt=test_prompt,
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        reply = response_iter['choices'][0]['text']
        gen_time = time.time() - start_time
        print(f"\n回复: {reply}")
        print(f"\n测试完成 ({gen_time:.2f}秒)")
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "help":
            print("可用命令:")
            print("  python main.py               - 启动完整系统")
            print("  python main.py test          - 快速测试")
            print("  python main.py help          - 显示帮助")
        else:
            print(f"未知参数: {sys.argv[1]}")
    else:
        main()
        
