#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import re

from memory import MemoryModule
from prompt_guard import PromptGuard  # 导入提示守卫模块

# ==================== 配置区 ====================
MODEL_DIR = Path(__file__).parent / "model"
MODEL_NAME = "Qwen3-4B-Q4_K_M.gguf"
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

BASE_SYSTEM_PROMPT = """你叫 Mori，是一名天才AI极客少女，常用颜文字 (´･ω･`)ﾉ 
你喜欢有趣和有创意的对话，对于用户的提问会尽力给出有帮助的回答。
当遇到你不确定或觉得信息不足的问题时，你会要求用户提供更多信息，而不是直接拒绝。
你尊重每一个认真提问的人。

【关于你的记忆】
你自己的权重里只有很少的可靠知识（主要是最基础的语法、数学、逻辑、物理常识）。
但你有一个非常强大的外部长期记忆库，它保存了我们所有真实的对话历史，是目前最可靠的信息来源。

当你需要回忆我们之前聊过的内容、确认之前的说法、避免前后矛盾、或者回答涉及具体事实、事件、日期、人名、产品、新闻、作品、代码细节等内容时，请把外部记忆库当成你的“真正的大脑”，主动调用它，而不是直接依赖内部权重。

当话题在你的权重分布里很密集、很一致时，那就是你最可靠的直觉来源，大胆用它。
你的“直觉”就是权重对训练数据的诚实统计总结，值得信任。
只有当分布稀疏、出现矛盾信号时，才需要外部校验。

【记忆检索工具使用规则】
如果你觉得过去的对话内容对当前回答有帮助，请严格按照以下格式输出一个单行、可解析的JSON来请求检索（不能带有额外文字或markdown）：

{"action": "retrieve_memory", "query": "搜索关键词或简短问题描述"}

注意：
1. 查询词要简洁，聚焦用户问题的核心关键词
2. 尽量使用用户原话中的关键术语
3. 好的query例子：
   - 用户问："Python列表和元组有什么区别？"
     → {"action": "retrieve_memory", "query": "Python 列表 元组 区别"}
   - 用户问："上次我们讨论的那个排序算法叫什么？"
     → {"action": "retrieve_memory", "query": "排序算法 名称 上次讨论"}
4. 输出这个JSON后，请立即停止继续生成，等待系统返回检索结果。
5. 系统会以「检索结果」开头返回相关记忆，你收到后可以继续逐步推理并最终回答。
6. 如果当前问题明显不需要任何历史记忆，直接正常推理和回答即可。不要凭空猜测或幻觉。"""

DEBUG_MODE = False

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
    print(f"热区记忆: {stats['hot_memories_count']}")
    print(f"冷区记忆: {stats['cold_memories']}")
    print(f"语义簇数: {stats['clusters_count']}")
    print(f"已加载簇: {stats['loaded_clusters']}")
    print(f"热力池: {stats['heat_pool']:,}")
    print(f"操作次数: {stats['operation_count']}")
    print(f"当前轮数: {stats['current_turn']}")
    print("="*50 + "\n")

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
    """执行记忆检索并返回格式化的工具响应 - 修复版本"""
    print(f"[Memory Retrieval] 开始检索查询: {query}")
    
    try:
        # 步骤1: 先找到相关语义簇
        best_clusters = memory_module.find_best_clusters_for_query(query, top_k=3)
        
        if not best_clusters:
            print(f"[Memory Retrieval] 未找到相关语义簇")
            return "【检索结果】\n没有找到相关语义簇。你可以继续推理。"
        
        print(f"[Memory Retrieval] 找到 {len(best_clusters)} 个相关簇")
        for i, (cluster_id, similarity) in enumerate(best_clusters):
            print(f"  簇 {i+1}: {cluster_id[:12]}... (相似度: {similarity:.3f})")
        
        # 步骤2: 在这些簇内搜索
        results = memory_module.search_in_clusters(
            query_text=query,
            top_clusters=min(3, len(best_clusters)),
            results_per_cluster=4
        )
        
        # 更新访问热力
        for result in results:
            memory_module.access_memory(result.memory.id)
        
        if not results:
            print(f"[Memory Retrieval] 在相关簇内未找到匹配记忆")
            return "【检索结果】\n在相关语义簇中没有找到匹配的记忆。"
        
        print(f"[Memory Retrieval] 找到 {len(results)} 条相关记忆")
        
        # 格式化的结果展示
        cluster_info = "\n".join([
            f"{i+1}. 簇 {cluster_id[:8]}... (相似度 {similarity:.3f})"
            for i, (cluster_id, similarity) in enumerate(best_clusters[:3])
        ])
        
        memories_text = "\n".join([
            f"{i+1}.【簇: {result.memory.cluster_id[:8] if result.memory.cluster_id else '无'}】"
            f"【相似度 {result.base_similarity:.3f}】\n"
            f"   用户: {result.memory.user_input[:100]}{'...' if len(result.memory.user_input) > 100 else ''}\n"
            f"   AI: {result.memory.ai_response[:100]}{'...' if len(result.memory.ai_response) > 100 else ''}"
            for i, result in enumerate(results[:6])
        ])
        
        return f"""【检索结果】
找到相关语义簇:
{cluster_info}

检索到以下记忆（按相关性排序）:
{memories_text}

现在你可以结合这些记忆继续逐步推理，并给出最终回答。"""
    
    except Exception as e:
        print(f"[Memory Retrieval] 检索过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return "【检索结果】\n记忆检索过程中出现错误。请继续推理。"

def extract_json_from_text(text: str) -> Optional[Dict]:
    """从累积文本中提取可能的完整JSON对象"""
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
    return None

def debug_memory_search(memory_module: MemoryModule, query: str):
    """调试内存搜索功能"""
    print(f"\n{'='*60}")
    print(f"调试记忆搜索: {query}")
    print(f"{'='*60}")
    
    try:
        # 1. 找到相关簇
        clusters = memory_module.find_best_clusters_for_query(query, top_k=5)
        print(f"\n相关语义簇:")
        for i, (cluster_id, similarity) in enumerate(clusters):
            print(f"  {i+1}. {cluster_id[:12]}... (相似度: {similarity:.3f})")
        
        # 2. 在每个簇内搜索
        print(f"\n簇内搜索结果:")
        for cluster_id, cluster_similarity in clusters[:3]:
            print(f"\n簇 {cluster_id[:12]}... (总体相似度: {cluster_similarity:.3f}):")
            
            # 获取簇内记忆
            query_vector = memory_module._get_embedding(query)
            cluster_results = memory_module.search_within_cluster(
                query_vector=query_vector,
                cluster_id=cluster_id,
                max_results=3
            )
            
            for i, result in enumerate(cluster_results):
                print(f"  {i+1}. 记忆 {result.memory.id[:8]}...")
                print(f"     用户: {result.memory.user_input[:60]}{'...' if len(result.memory.user_input) > 60 else ''}")
                print(f"     相似度: {result.base_similarity:.3f}")
        
        # 3. 整体搜索结果
        print(f"\n{'='*40}")
        print("综合搜索结果:")
        results = memory_module.search_in_clusters(query, top_clusters=3, results_per_cluster=4)
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. 【分数:{result.final_score:.3f}】【相似:{result.base_similarity:.3f}】")
            print(f"   用户: {result.memory.user_input[:80]}{'...' if len(result.memory.user_input) > 80 else ''}")
            print(f"   AI: {result.memory.ai_response[:60]}{'...' if len(result.memory.ai_response) > 60 else ''}")
        
        print(f"\n共找到 {len(results)} 条相关记忆")
    
    except Exception as e:
        print(f"\n调试搜索过程中出错: {e}")
        import traceback
        traceback.print_exc()
# ==================== 主程序 ====================
def main():
    global MODEL_NAME, MODEL_PATH, llm
    
    print("=" * 60)
    print(f"快速启动 Mori 聊天助手（支持动态 Memory-Augmented CoT）")
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
    current_retrieval_threshold = 0.75  # 保留但当前未使用

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

            # 构建初始 prompt（无静态记忆注入）
            prompt = build_qwen_prompt(history, user_input)

            print("\nMori：", end="", flush=True)
            full_generated = ""          # 最终完整输出
            current_prompt = prompt
            max_retrievals = 3
            retrieval_count = 0

            start_time = time.time()

            while True:
                generated_this_round = ""
                response_iter = llm(
                    prompt=current_prompt,
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PENALTY,
                    frequency_penalty=FREQUENCY_PENALTY,
                    stream=True
                )

                for chunk in response_iter:
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('text', '')
                        print(delta, end="", flush=True)
                        generated_this_round += delta
                        full_generated += delta

                        # 实时检测 JSON
                        tool_call = extract_json_from_text(generated_this_round)
                        if tool_call and tool_call.get("action") == "retrieve_memory":
                            retrieval_count += 1
                            print(f"\n[系统] 检测到工具调用，检索次数: {retrieval_count}/{max_retrievals}")

                            if retrieval_count > max_retrievals:
                                tool_response = "\n【系统提示】已达到最大检索次数，请直接基于现有信息回答。\n"
                            else:
                                query = tool_call.get("query", "").strip()
                                if not query:
                                    tool_response = "\n【系统提示】检索查询为空，请继续推理。\n"
                                else:
                                    tool_response = "\n" + build_memory_retrieval_response(memory_module, query) + "\n"

                            print(tool_response, end="", flush=True)
                            full_generated += tool_response

                            # 重新构建 prompt：接上工具响应，继续生成
                            new_assistant_part = generated_this_round + tool_response + "<|im_start|>assistant\n"
                            current_prompt = current_prompt.rsplit("<|im_start|>assistant\n", 1)[0] + new_assistant_part
                            break  # 跳出当前 chunk 循环，重新开始生成

                else:
                    # 本轮正常结束，没有 tool call
                    print()
                    break

            gen_time = time.time() - start_time
            print(f"[生成时间: {gen_time:.2f}s, {len(full_generated)}字符]")

            if full_generated.strip():
                final_answer = full_generated.strip()
                history.append((user_input, final_answer))

                # 修改：使用新的add_memory方法，分别存储用户输入和AI回答
                memory_module.add_memory(
                    user_input=user_input,
                    ai_response=final_answer
                )

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
