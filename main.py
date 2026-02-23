#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random
import logging
import hashlib
import re
import queue
import threading
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

from memory_system import MemoryModule
from tool import ToolConfig, ToolCallManager, handle_tool_call  # 导入工具模块

# ==================== 配置区 ====================
MODEL_DIR = Path(__file__).parent / "model"
MODEL_NAME = "gpt-oss-20b-UD-Q6_K_XL.gguf"
MODEL_PATH = MODEL_DIR / MODEL_NAME

SMALL_MODEL_DIR = MODEL_DIR
SMALL_MODEL_NAME = "Qwen3-4B-Q4_K_M.gguf"
SMALL_MODEL_THREADS = 20
SMALL_MODEL_PATH = SMALL_MODEL_DIR / SMALL_MODEL_NAME
USE_SMALL_MODEL_FOR_FACT_EXTRACTION = True

CHAT_FORMAT = "auto"

ENABLE_SLIDING_WINDOW = True
RESERVED_TOKENS = 1024
PER_MESSAGE_OVERHEAD = 4

ATOMIC_PENDING_LIMIT_FACTOR = 1.0
ATOMIC_WAIT_RATIO = 0.5

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

print(f"主模型文件: {MODEL_PATH}")
if MODEL_PATH.exists():
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"主模型大小: {size_mb:.2f} MB")

if USE_SMALL_MODEL_FOR_FACT_EXTRACTION:
    print(f"小模型文件: {SMALL_MODEL_PATH}")
    if SMALL_MODEL_PATH.exists():
        size_mb = SMALL_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"小模型大小: {size_mb:.2f} MB")
    else:
        print("警告：小模型文件不存在，将使用规则提取事实")

EMBEDDING_MODEL_PATH = "./model/Qwen3-Embedding-0.6B/"
EMBEDDING_DIM = 1024

MAX_CONTEXT_TOKENS = 8192
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPEAT_PENALTY = 1.1
FREQUENCY_PENALTY = 0.3

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
你的输出：{"action": "retrieve_memory", "query": "推荐的书名"}

用户：我们之前讨论过的Python代码怎么写的？
你的输出：{"action": "retrieve_memory", "query": "用户讨论的python代码"}

用户：我不记得那个API的参数了，你记得吗？
你的输出：{"action": "retrieve_memory", "query": "API参数"}

用户：今天天气怎么样？
你的输出：(直接回答天气问题，不需要调用工具)

【注意】
1. 只有在确实需要回忆过去信息时才输出JSON。
2. 如果是常识问题或新话题，直接正常回答。
3. 输出JSON后立即停止生成，等待系统返回结果。
4. 在最终回答时，使用中文。"""

DEBUG_MODE = False

MEMORY_TRIGGER_WORDS = [
    "上次", "之前", "以前", "记得", "还记得", "忘了", "忘记", 
    "我们聊过", "我们说过", "讨论过", "提到过", 
    "那个...", "历史", "回忆", "回忆一下",
    "last time", "remember", "mentioned before"
]

# ==================== 原子事实抽取函数（增强版）====================
def extract_atomic_facts(user_input: str, ai_response: str) -> List[str]:
    """将用户输入和AI响应拆分为原子事实，优先使用小模型生成"""
    # 优先使用小模型提取（如果启用且已加载）
    if USE_SMALL_MODEL_FOR_FACT_EXTRACTION and small_llm is not None:
        try:
            cleaned_ai = remove_cot_content(ai_response)
            cleaned_ai = extract_final_response(cleaned_ai).strip()
            if not cleaned_ai:
                cleaned_ai = ai_response.strip()

            system_prompt = """你是“原子事实抽取器”。

你的任务是：
从【用户】和【AI】的对话中，抽取可以长期存储的“原子事实”。

【什么是原子事实？】
- 单条、独立、可检索的信息
- 去除语气词、寒暄、重复
- 不要复述整段对话
- 不要写解释
- 不要写总结
- 不要写规则
- 不要写“用户说”或“AI说”

【必须遵守】
1. 每条事实单独一行
2. 每条不超过25个字
3. 使用陈述句
4. 不要包含“用户”“AI”
5. 不要重复输入原句
6. 不要输出空行
7. 如果没有值得存储的事实，输出：无

【示例】

输入：
[用户]
我昨天写heatmem写到凌晨两点。
[AI]
你真的很拼。

输出：
用户熬夜写heatmem
写到凌晨两点

----

输入：
[用户]
今天天气不错。
[AI]
是的，很晴朗。

输出：
无

----

现在开始抽取："""  # 原有prompt不变

            user_content = f"[用户]\n{user_input}\n[AI]\n{cleaned_ai}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = small_llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            
            output = response['choices'][0]['message']['content'].strip()
            output = remove_cot_content(output)
            print(f"[系统] 事实提取结果:{output}")
            raw_facts = [line.strip() for line in output.split('\n') if line.strip()]
            
            facts = []
            for fact in raw_facts:
                fact = re.sub(r'^(用户|AI|assistant|user)[\s:：]*', '', fact, flags=re.IGNORECASE).strip()
                fact = re.sub(r'^"|"$', '', fact)
                if fact.startswith("规则") or "提取" in fact or "事实" in fact:
                    continue
                if "开始提取" in fact or "对话：" in fact:
                    continue
                if len(fact) < 5:
                    continue
                if fact in ["用户", "AI", "[用户]", "[AI]"]:
                    continue
                facts.append(fact)
            
            unique_facts = []
            seen = set()
            for fact in facts:
                key = fact.strip().lower()
                key = re.sub(r'[，。！？、；：""''（）]', '', key)
                if key not in seen:
                    seen.add(key)
                    unique_facts.append(fact)
            
            if unique_facts:
                print(f"[系统] 小模型提取到 {len(unique_facts)} 条事实")
                return unique_facts
            else:
                print("[系统] 小模型未返回有效事实，不存储任何事实")
                return []  # 直接返回空列表，不使用规则回退
                
        except Exception as e:
            print(f"[系统] 小模型提取事实失败: {e}")
            import traceback
            traceback.print_exc()
            return []  # 发生异常也返回空列表，不使用规则回退
    
    # 如果小模型未启用或未加载，直接返回空列表
    print("[系统] 小模型未启用或未加载，不提取事实")
    return []

# ==================== 原子事实处理队列 ====================
atomic_extract_queue = queue.Queue()   # 原始对话 -> 后台提取线程
atomic_facts_queue = queue.Queue()     # 提取结果 -> 主线程存储
_stop_atomic_worker = False

def atomic_worker():
    """后台线程：不断从队列中取出原始对话，提取原子事实，将结果放入 facts_queue"""
    while not _stop_atomic_worker:
        try:
            task = atomic_extract_queue.get(timeout=1)
            if task is None:   # 退出信号
                break
            user_input, ai_response, turn_count = task
            facts = extract_atomic_facts(user_input, ai_response)
            if facts:
                atomic_facts_queue.put((turn_count, user_input, ai_response, facts))
            atomic_extract_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[原子事实后台线程] 错误: {e}")

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
            self.embedding_model = SentenceTransformer(model_path, device='cpu')
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
small_llm = None

# ==================== 模型加载函数 ====================
def load_llama_model():
    global llm
    try:
        from llama_cpp import Llama
        print(f"正在加载模型: {MODEL_PATH.name}")
        print(f"模型大小: {MODEL_PATH.stat().st_size / (1024**3):.2f} GB")
        start_time = time.time()
        
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        free_memory = int(result.stdout.strip().split('\n')[0])
        print(f"可用显存: {free_memory} MiB")
        
        load_kwargs = {
            "model_path": str(MODEL_PATH),
            "n_ctx": MAX_CONTEXT_TOKENS,
            "n_threads": 8,
            "n_threads_batch": 8,
            "n_gpu_layers": 99,
            "main_gpu": 0,
            "tensor_split": None,
            "n_gpu_layers_experts": 99,
            "experts_per_gpu": 128,
            "expert_used_count": 8,
            "use_mmap": False,
            "use_mlock": False,
            "low_vram": False,
            "n_batch": 1024,
            "n_ubatch": 512,
            "batch_threads": 8,
            "embedding": False,
            "verbose": True,
            "logits_all": False,
        }
        
        if free_memory > 20000:
            load_kwargs["n_batch"] = 2048
            load_kwargs["n_ubatch"] = 1024
            print("显存充足，启用大batch模式")
        
        print("\n=== GPU 分配策略 ===")
        print(f"总层数: {load_kwargs['n_gpu_layers']}")
        print(f"专家层全部在 GPU: 是")
        print(f"专家总数: {load_kwargs['experts_per_gpu']}")
        print(f"每个token使用的专家: {load_kwargs['expert_used_count']}")
        print(f"Batch size: {load_kwargs['n_batch']}")
        print("===================\n")
        
        if CHAT_FORMAT != "auto":
            load_kwargs["chat_format"] = CHAT_FORMAT
        
        print("开始加载模型（GPU + 专家层优化）...")
        llm = Llama(**load_kwargs)
        
        load_time = time.time() - start_time
        print(f"✓ 主模型加载完成 ({load_time:.2f}秒)")
        
        if hasattr(llm, 'n_gpu_layers'):
            print(f"✓ GPU层数: {llm.n_gpu_layers}")
        if hasattr(llm, 'model_metadata'):
            print(f"✓ 专家层在GPU: {llm.model_metadata.get('experts_on_gpu', '未知')}")
        
        return True
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_small_model():
    global small_llm
    if not USE_SMALL_MODEL_FOR_FACT_EXTRACTION:
        print("[信息] 小模型功能已禁用")
        return False
    if not SMALL_MODEL_PATH.exists():
        print(f"[警告] 小模型文件不存在: {SMALL_MODEL_PATH}")
        return False
    
    try:
        from llama_cpp import Llama
        print(f"\n正在加载小模型: {SMALL_MODEL_PATH.name}")
        print(f"小模型大小: {SMALL_MODEL_PATH.stat().st_size / (1024**3):.2f} GB")
        start_time = time.time()
        
        load_kwargs = {
            "model_path": str(SMALL_MODEL_PATH),
            "n_ctx": 512,
            "n_threads": 4,
            "n_gpu_layers": 0,
            "verbose": False,
            "n_threads": SMALL_MODEL_THREADS,
            "chat_format": "chatml",
        }
        
        small_llm = Llama(**load_kwargs)
        load_time = time.time() - start_time
        print(f"✓ 小模型加载完成 ({load_time:.2f}秒)，使用 {SMALL_MODEL_THREADS} 线程")
        return True
    except Exception as e:
        print(f"[警告] 小模型加载失败: {e}")
        small_llm = None
        return False

# ==================== 工具函数 ====================
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
    print(f"当前轮数: {stats['current_turn']}")
    print("="*50 + "\n")

def print_dialogue_stats(dialogue_manager):
    """打印原始对话统计"""
    stats = dialogue_manager.get_stats()
    print("\n" + "="*50)
    print("原始对话统计:")
    print(f"总对话轮数: {stats['total_lines']}")
    print(f"起始轮数: {stats['first_turn']}")
    print(f"最新轮数: {stats['last_turn']}")
    print("="*50 + "\n")

def remove_cot_content(text: str) -> str:
    if not text:
        return text
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    if '<think>' in cleaned:
        think_index = cleaned.find('<think>')
        cleaned = cleaned[:think_index]
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()
    return cleaned

def extract_final_response(text: str) -> str:
    marker = "<|channel|>final<|message|>"
    if marker in text:
        pos = text.rfind(marker)
        after_marker = text[pos + len(marker):]
        after_marker = after_marker.lstrip()
        return after_marker
    else:
        return text

def print_model_info():
    model_info = model_manager.get_model_info()
    print("\n" + "="*50)
    print("模型信息:")
    print(f"嵌入模型已加载: {model_info['embedding_model_loaded']}")
    print(f"嵌入模型路径: {model_info['embedding_model_path']}")
    print(f"嵌入维度: {model_info['embedding_dim']}")
    print(f"模型类型: {model_info['model_type']}")
    print(f"主语言模型: {MODEL_NAME}")
    if llm and hasattr(llm, 'chat_format'):
        print(f"主模型聊天格式: {llm.chat_format}")
    if small_llm is not None:
        print(f"小模型（事实提取）: {SMALL_MODEL_NAME} (已加载，CPU运行)")
    else:
        print(f"小模型（事实提取）: 未加载")
    print("="*50 + "\n")

def count_tokens(text: str) -> int:
    global llm
    if llm is None:
        return len(text) // 2
    try:
        tokens = llm.tokenize(text.encode('utf-8'))
        return len(tokens)
    except Exception as e:
        print(f"[Token计数] 失败: {e}")
        return len(text) // 2

def trim_messages(messages: list, max_total_tokens: int) -> tuple[list, int]:
    if not ENABLE_SLIDING_WINDOW:
        non_system = [msg for msg in messages if msg["role"] != "system"]
        if non_system and non_system[-1]["role"] == "user":
            full_rounds = (len(non_system) - 1) // 2
        else:
            full_rounds = len(non_system) // 2
        return messages, full_rounds

    system_msgs = [msg for msg in messages if msg["role"] == "system"]
    other_msgs = [msg for msg in messages if msg["role"] != "system"]

    system_tokens = sum(count_tokens(msg["content"]) for msg in system_msgs)

    accumulated = 0
    keep_indices = []
    for i in range(len(other_msgs) - 1, -1, -1):
        msg = other_msgs[i]
        msg_tokens = count_tokens(msg["content"]) + PER_MESSAGE_OVERHEAD
        if accumulated + msg_tokens + system_tokens <= max_total_tokens:
            accumulated += msg_tokens
            keep_indices.append(i)
        else:
            break

    keep_indices.sort()
    kept_other_msgs = [other_msgs[i] for i in keep_indices]

    trimmed = system_msgs + kept_other_msgs
    if kept_other_msgs and kept_other_msgs[-1]["role"] == "user":
        full_rounds = (len(kept_other_msgs) - 1) // 2
    else:
        full_rounds = len(kept_other_msgs) // 2

    print(f"[滑动窗口] 原始消息数: {len(messages)}, 裁剪后: {len(trimmed)}, "
          f"估算 token: {system_tokens + accumulated}, 保留完整轮数: {full_rounds}")
    return trimmed, full_rounds

def debug_memory_search(memory_module: MemoryModule, query: str):
    print(f"\n{'='*60}")
    print(f"调试记忆搜索: {query}")
    print(f"{'='*60}")
    
    try:
        results = memory_module.search_original_memories(query_text=query, max_results=10)
        
        print(f"\n找到 {len(results)} 条相关原子事实:")
        for i, (mem, score) in enumerate(results):
            print(f"{i+1}. 【轮数:{mem.created_turn}】【相似度:{score:.3f}】")
            print(f"   事实: {mem.user_input[:80]}{'...' if len(mem.user_input) > 80 else ''}")
        
        print(f"\n原子事实统计:")
        print(f"  原子事实数: {len(results)}")
    
    except Exception as e:
        print(f"\n调试搜索过程中出错: {e}")
        import traceback
        traceback.print_exc()

def debug_dialogue_search(dialogue_manager, query: str):
    """调试原始对话搜索"""
    print(f"\n{'='*60}")
    print(f"调试原始对话搜索: {query}")
    print(f"{'='*60}")
    
    results = dialogue_manager.search_by_keyword(query, max_results=10)
    
    print(f"\n找到 {len(results)} 条相关原始对话:")
    for i, (turn, user_input, ai_response) in enumerate(results):
        print(f"{i+1}. 【轮数:{turn}】")
        print(f"   用户: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
        print(f"   AI: {ai_response[:60]}{'...' if len(ai_response) > 60 else ''}")

# ==================== 主线程处理原子事实队列 ====================
def process_atomic_facts_queue(memory_module: MemoryModule):
    """从队列中取出所有待存储的原子事实并存入记忆模块"""
    while True:
        try:
            turn, user_input, ai_response, facts = atomic_facts_queue.get_nowait()
            memory_module.add_atomic_memories(
                facts=facts,
                user_input=user_input,
                ai_response=ai_response,
                metadata={"source": "conversation", "turn": turn}
            )
            atomic_facts_queue.task_done()
        except queue.Empty:
            break

# ==================== 话题概括生成函数（使用小模型）====================
def generate_topic_summary(prompt: str) -> str:
    """使用小模型生成话题概括"""
    if small_llm is None:
        return ""
    try:
        messages = [
            {"role": "system", "content": "你是一个话题概括助手，根据对话内容生成一句话概括。/no_think"},
            {"role": "user", "content": prompt}
        ]
        response = small_llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
        )
        summary = response['choices'][0]['message']['content'].strip()
        summary = remove_cot_content(summary)
        
        # ========== 新增打印 ==========
        print(f"[小模型原始输出] {summary}")
        # =============================
        
        return summary
    except Exception as e:
        print(f"[概括生成失败] {e}")
        return ""

# ==================== 主程序 ====================
def main():
    global MODEL_NAME, MODEL_PATH, llm, small_llm
    
    print("=" * 60)
    print(f"快速启动 Mori 聊天助手（支持动态 Memory-Augmented CoT + 原子事实记忆 + 话题分割）")
    print(f"使用模型: {MODEL_NAME}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"聊天格式: {CHAT_FORMAT}")
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
    
    print("\n[1/4] 加载主语言模型...")
    if not load_llama_model():
        print("主模型加载失败，退出程序")
        sys.exit(1)
    
    print("[2/4] 加载小模型（原子事实提取）...")
    load_small_model()
    
    print("[3/4] 初始化全局嵌入模型...")
    if not model_manager.load_embedding_model(EMBEDDING_MODEL_PATH):
        print("警告：嵌入模型加载失败，使用哈希嵌入")
    
    # ========== 启用 SQLite WAL 模式 ==========
    try:
        conn = sqlite3.connect("memory/memory.db")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
        print("[系统] 数据库已启用 WAL 模式")
    except Exception as e:
        print(f"[警告] 无法设置数据库 WAL 模式: {e}")
    
    # ========== 初始化主记忆模块 ==========
    print("[4/4] 初始化主记忆模块...")
    memory_module_main = MemoryModule(
        embedding_func=model_manager.get_embedding,
        similarity_func=model_manager.compute_similarity,
        small_llm_func=generate_topic_summary
    )
    
    # 获取对话管理器引用（方便使用）
    dialogue_manager = memory_module_main.dialogue_manager
    topic_segmenter = memory_module_main.topic_segmenter
    
    # 初始化工具调用管理器
    tool_config = ToolConfig.from_list(
        trigger_words=MEMORY_TRIGGER_WORDS,
        max_retrievals=3
    )
    tool_manager = ToolCallManager(tool_config)
    
    # 启动后台提取线程
    atomic_thread = threading.Thread(target=atomic_worker, daemon=True)
    atomic_thread.start()
    
    print("\n" + "=" * 60)
    print("系统准备就绪！可以开始对话")
    print("=" * 60)
    print("\n可用命令: quit / exit / q / stats / dialogue_stats / model_info / save / clear / history / model / debug_memory / debug_dialogue")
    print("-" * 50)

    # 修改：history 存储格式改为 (global_turn, user_input, ai_response)
    history: List[Tuple[int, str, str]] = []

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
                print_memory_stats(memory_module_main)
                continue
            elif user_input.lower() == "dialogue_stats":
                print_dialogue_stats(dialogue_manager)
                continue
            elif user_input.lower() == "model_info":
                print_model_info()
                continue
            elif user_input.lower() == "clear":
                history.clear()
                print("对话历史已清空")
                continue
            elif user_input.lower() == "history":
                print(f"\n当前对话历史（{len(history)} 轮）：")
                for i, (turn, u, a) in enumerate(history, 1):
                    print(f"{i}. 【轮次 {turn}】你：{u[:50]}{'...' if len(u)>50 else ''}")
                    print(f"   Mori：{a[:50]}{'...' if len(a)>50 else ''}")
                print()
                continue
            elif user_input.lower() == "model":
                print(f"\n语言模型信息:")
                print(f"  名称: {MODEL_NAME}")
                print(f"  路径: {MODEL_PATH}")
                print(f"  大小: {MODEL_PATH.stat().st_size / (1024**3):.2f} GB")
                print(f"  上下文长度: {MAX_CONTEXT_TOKENS} tokens")
                if llm and hasattr(llm, 'chat_format'):
                    print(f"  聊天格式: {llm.chat_format}")
                continue
            elif user_input.lower() == "debug_memory":
                query = input("请输入搜索查询: ").strip()
                if query:
                    debug_memory_search(memory_module_main, query)
                continue
            elif user_input.lower() == "debug_dialogue":
                query = input("请输入搜索查询: ").strip()
                if query:
                    debug_dialogue_search(dialogue_manager, query)
                continue
                
            if not user_input:
                continue

            # ==================== 核心生成逻辑 ====================
            
            # 重置工具管理器
            tool_manager.reset()
            
            old_history_len = len(history)   # 记录裁剪前的历史轮数
            
            messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
            for turn, user_msg, ai_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": ai_msg})
            messages.append({"role": "user", "content": user_input})
            
            if ENABLE_SLIDING_WINDOW:
                max_history_tokens = MAX_CONTEXT_TOKENS - RESERVED_TOKENS
                messages, kept_rounds = trim_messages(messages, max_history_tokens)
            else:
                non_system = [msg for msg in messages if msg["role"] != "system"]
                if non_system and non_system[-1]["role"] == "user":
                    kept_rounds = (len(non_system) - 1) // 2
                else:
                    kept_rounds = len(non_system) // 2
            
            # ========== 计算裁剪掉的历史轮次范围，并输出话题信息 ==========
            cutoff = old_history_len - kept_rounds
            if cutoff > 0:
                # 获取被裁剪掉的全局 turn 范围
                cut_turns = [turn for turn, _, _ in history[:cutoff]]
                if cut_turns:
                    min_cut_turn = min(cut_turns)
                    max_cut_turn = max(cut_turns)
                    print(f"[系统] 由于上下文限制，裁剪掉了第 {min_cut_turn}~{max_cut_turn} 轮对话")
                    # 获取被裁剪轮次范围内的话题段
                    ranges = topic_segmenter.get_topics_in_range(min_cut_turn, max_cut_turn)
                    for start, end in ranges:
                        summary = topic_segmenter.get_summary_for_topic(start, end)
                        if summary:
                            print(f"  话题 {start}-{end} 概括：{summary}")
                        else:
                            print(f"  话题 {start}-{end} (未概括)")
            
            # 判断是否强制检索
            forced_mode = tool_manager.should_force_retrieval(user_input)
            if forced_mode:
                messages.append({"role": "assistant", "content": '{"action": "retrieve_memory", "query": "'})

            print("\nMori：", end="", flush=True)
            
            full_response = ""
            tool_call_detected = False
            tool_query = ""
            start_time = time.time()
            
            try:
                current_seed = int(time.time() * 1000) % 2**32
                response_stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE if not forced_mode else 0.3,
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PENALTY,
                    frequency_penalty=FREQUENCY_PENALTY,
                    stream=True,
                    seed=current_seed
                )
                
                for chunk in response_stream:
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_response += delta
                    
                    if not tool_call_detected:
                        # 使用工具管理器检测工具调用
                        detected, query = tool_manager.detect_in_stream(full_response, forced_mode)
                        if detected:
                            tool_call_detected = True
                            tool_query = query
                            break
                
                # 处理工具调用
                if tool_call_detected and tool_manager.can_retry():
                    final_answer, new_count = handle_tool_call(
                        llm=llm,
                        messages=messages,
                        full_response=full_response,
                        tool_query=tool_query,
                        retrieval_count=tool_manager.get_count(),
                        max_retrievals=tool_manager.config.max_retrievals,
                        memory_module=memory_module_main,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        repeat_penalty=REPEAT_PENALTY,
                        frequency_penalty=FREQUENCY_PENALTY,
                        max_new_tokens=MAX_NEW_TOKENS
                    )
                    tool_manager.increment_count()
                    full_response = final_answer
                else:
                    full_response = full_response.strip()
                
                print()
                
            except Exception as e:
                print(f"\n[系统] 生成过程中出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            gen_time = time.time() - start_time
            print(f"[生成时间: {gen_time:.2f}s, {len(full_response)}字符]")
            
            if full_response:
                cleaned_response = extract_final_response(full_response)
                
                # ========== 核心修改：使用数据库的全局 turn ==========
                # 递增全局对话轮次，获取本轮 turn
                current_global_turn = memory_module_main.increment_turn()
                
                # ===== 1. 记录原始对话到 history.txt =====
                dialogue_manager.add_dialogue(
                    turn=current_global_turn,
                    user_input=user_input,
                    ai_response=cleaned_response
                )
                
                # ===== 2. 获取用户向量并用于话题分割 =====
                user_vector = model_manager.get_embedding(user_input)
                topic_segmenter.add_turn_vector(current_global_turn, user_vector)
                
                # ===== 3. 记录到内存历史列表（使用全局 turn） =====
                history.append((current_global_turn, user_input, cleaned_response))
                
                # ===== 4. 原子事实提取（异步） =====
                pending = atomic_extract_queue.qsize()
                window_len = kept_rounds
                if window_len > 0:
                    limit = int(window_len * ATOMIC_PENDING_LIMIT_FACTOR)
                    target = int(window_len * ATOMIC_WAIT_RATIO)
                    if pending > limit:
                        print(f"[积压控制] 当前积压任务数 {pending} 超过限制 {limit}，等待降至 {target}...")
                        while atomic_extract_queue.qsize() > target:
                            time.sleep(1)
                atomic_extract_queue.put((user_input, cleaned_response, current_global_turn))
                print(f"[系统] 原子事实提取任务已提交到后台（轮次 {current_global_turn}）")
                
                # ===== 5. 处理已提取好的原子事实 =====
                process_atomic_facts_queue(memory_module_main)

                if len(history) % 10 == 0:
                    print_memory_stats(memory_module_main)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出...")
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        global _stop_atomic_worker
        _stop_atomic_worker = True
        atomic_extract_queue.join()
        atomic_thread.join(timeout=5)
        
        # 处理剩余未存储的原子事实
        process_atomic_facts_queue(memory_module_main)
        atomic_facts_queue.join()
        
        # 确保最后一个话题被处理
        topic_segmenter.finalize_topics()
        
        try:
            memory_module_main.cleanup()
        except Exception as e:
            print(f"清理主记忆模块时出错: {e}")
        
        print("\n最终统计:")
        print_model_info()
        print_dialogue_stats(dialogue_manager)
        print("程序结束")

def quick_test():
    print("=" * 60)
    print("快速测试模式")
    print("=" * 60)
    
    if not load_llama_model():
        return
    
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，请简单介绍一下自己。"}
    ]
    
    print("测试消息:")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print("\n正在生成回复...")
    
    try:
        start_time = time.time()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        reply = response['choices'][0]['message']['content']
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