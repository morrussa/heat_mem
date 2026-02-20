# Heat Mem – 热力记忆系统

**Heat Mem** 是一个为 AI 对话系统设计的长期记忆模块，通过**热力值**动态管理记忆的重要性，并结合**语义聚类**与**联想图（Waypoint）** 技术，让 AI 能够高效地存储、检索和联想历史对话。

默认**main.py**搭载了一个名为mori的agent。没别的意思，单纯提醒你再真实，它也是死的。

---

## ✨ 核心特性

- **🔥 纯热力驱动的记忆生命周期**  
  每个记忆和语义簇都拥有热力值，系统通过全局热力池自动分配、回收和调节热力，使重要记忆保持活跃，冷门记忆自然沉降，实现长期记忆的自动管理。对，它没有时间衰减。

- **🧠 语义聚类（Semantic Clustering）**  
  基于用户输入的向量表示，将相似的记忆自动聚合成簇，支持 Annoy 快速索引，大幅提升大规模记忆下的检索效率。

- **🔗 联想图（Waypoint）**  
  在记忆之间建立有向边，通过共现、语义相似度等方式构建联想关系，支持多跳扩散检索，让 AI 能够“想起”间接相关的信息。

- **📊 分层搜索（Layered Search）**  
  将相似度划分为多个区间，结合热力权重、访问频率和新鲜度奖励，从不同层次返回结果，兼顾相关性与多样性。

- **🔄 原子事实存储与原始对话聚合**  
  将对话拆解为原子事实独立存储，通过 `parent_turn` 关联回原始对话，既保留了细节又便于溯源。

- **📜 历史记录管理器**  
  基于轮次（turn）记录所有原始对话，支持按轮次、范围、关键词检索，并提供 LRU 缓存与磁盘持久化。

- **✅ 事务与一致性保证**  
  提供 STRONG / IMMEDIATE / EVENTUAL 三种一致性级别，确保热力、簇、记忆之间的数据一致性。

- **⚡ 多级缓存优化**  
  向量缓存、相似度缓存、权重缓存、簇搜索缓存，大幅减少重复计算，提升响应速度。

- **🛠️ 完善的配置系统**  
  所有参数均可通过 `Config` 类调整，适应不同规模的应用场景。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/yourname/heat-mem.git
cd heat-mem
pip install -r requirements.txt
```

**依赖**：  
- Python ≥ 3.8  
- numpy  
- annoy
- sentence-transformers

### 2. 初始化记忆模块

```python
from heat_mem.core import MemoryModule

# 使用默认配置
mem = MemoryModule()

# 或者传入自定义的嵌入函数
def my_embedding(text: str) -> np.ndarray:
    # 使用你自己的模型生成向量
    return ...

mem = MemoryModule(embedding_func=my_embedding)
```

### 3. 添加记忆

```python
# 添加一次对话（原子事实会被自动提取，你需要自行实现提取逻辑）
memory_id = mem.add_memory(
    user_input="你喜欢什么颜色？",
    ai_response="我喜欢蓝色，像大海一样。",
    metadata={"session_id": "123"}
)

# 或者直接存储多个原子事实
facts = ["Mori喜欢蓝色", "蓝色让人联想到大海"]
mem.add_atomic_memories(
    facts=facts,
    user_input="你喜欢什么颜色？",
    ai_response="我喜欢蓝色，像大海一样。",
    metadata={"turn": 42}
)
```

### 4. 搜索记忆

```python
# 基本相似度搜索
results = mem.search_similar_memories(
    query_text="海洋的颜色",
    max_results=5
)
for r in results:
    print(f"{r.memory.summary} (得分: {r.final_score:.3f})")

# 分层搜索（获得更多样化的结果）
layered = mem.get_layered_search_results(
    query_text="海洋的颜色",
    flatten_results=True
)

# 搜索原始对话（聚合原子事实 + 联想扩散）
originals = mem.search_original_memories(
    query_text="海洋的颜色",
    max_results=5
)
for orig_mem, score in originals:
    print(f"[轮次 {orig_mem.created_turn}] 用户: {orig_mem.user_input}")
```

### 5. 手动触发联想（Waypoint）

```python
# 记录一轮对话中出现的所有记忆ID（用于建立共现边）
mem.record_turn_memories([mem_id1, mem_id2, mem_id3])

# 获取联想图统计
print(mem.get_waypoint_stats())
```

---

## ⚙️ 配置说明

所有配置项均位于 `heat_mem/config.py` 的 `Config` 类中。你可以通过修改该文件或实例化时传入自定义配置对象来调整系统行为。

关键配置示例：

```python
from heat_mem.config import Config

config = Config()
config.TOTAL_HEAT = 20_000_000          # 总热力池大小
config.CLUSTER_SIMILARITY_THRESHOLD = 0.85  # 簇相似度阈值
config.WAYPOINT_MAX_HOPS = 2            # 联想最大深度
config.LAYERED_SEARCH_MAX_TOTAL_RESULTS = 10

mem = MemoryModule(config=config)
```

详细配置项请参见 [config.py](heat_mem/config.py)。

---

## 📖 API 概览

| 方法 | 描述 |
|------|------|
| `add_memory(user_input, ai_response, metadata)` | 添加一次完整对话，自动提取原子事实（需自行实现提取逻辑） |
| `add_atomic_memories(facts, user_input, ai_response, metadata)` | 直接存储原子事实列表，处理重复检测与合并 |
| `search_similar_memories(query_text/vector, max_results)` | 基础相似度搜索 |
| `search_layered_memories(...)` | 分层搜索，返回各层结果 |
| `get_layered_search_results(...)` | 获取扁平化的分层搜索结果 |
| `search_original_memories(...)` | 聚合原子事实返回原始对话，并利用联想图扩散 |
| `search_within_cluster(cluster_id, ...)` | 在指定簇内搜索 |
| `find_best_clusters_for_query(query, top_k)` | 查找与查询最匹配的簇 |
| `access_memory(memory_id)` | 访问一条记忆（增加访问计数、更新热力） |
| `record_turn_memories(memory_ids)` | 记录一轮对话中出现的所有记忆ID，用于建立共现边 |
| `reinforce_waypoint_edges(seed_ids, hit_ids)` | 强化种子到扩散命中的边 |
| `get_stats()` | 获取系统统计信息（热力、缓存命中率、联想图规模等） |
| `cleanup()` | 关闭资源，保存状态 |

更多方法请参考 [core.py](heat_mem/core.py) 中的 `MemoryModule` 类。

---

## 🏗️ 架构概览

```
memory_system/
├── core.py                 # 核心 MemoryModule 与事务上下文
├── models.py               # 数据类（MemoryItem, SemanticCluster, WaypointEdge...）
├── config.py               # 全局配置
├── utils.py                # 工具函数（向量转换、相似度计算等）
├── infrastructure/
│   ├── database.py         # 数据库连接与初始化
│   ├── cache.py            # 多级缓存管理
│   ├── locking.py          # 分布式锁（线程级）
│   └── history.py          # 历史记录管理器
└── services/
    ├── heat_system.py      # 热力分配、回收、暂存热力处理
    ├── cluster_service.py  # 语义簇管理、Annoy索引
    ├── search_service.py   # 搜索逻辑（分层、簇内、原始对话联想）
    └── waypoint.py         # 联想图服务
```

---

## 📄 许可证

还没定，无所谓。

---

## 🌟 致谢和碎碎念

HeatMem的设计借鉴了openMemory项目的waypoint联想图。

我可以很自信的说这个项目是所有agent Memory中最癫的那个，因为我要求全量保存记忆的同时限制规模，你可能觉得很矛盾，但是它真的是这样的。

简单来说，在默认设置中，每个记忆写入时如果相似度>0.95，那么就追加原子事实对应的trun号。也就是说，拓扑上，它确实是有限的无限。

所以我完全放弃了记忆压缩和修剪，因为在我的认知中，任何形式的压缩都是对过去的背叛。所以我们不能要求LLM有自知力，论据：LLM（除diffusionLLM）只是单纯预测下一个token,如果让它自组织，即将记忆作为了LLM权重的附属品。如果你真的希望它活过来，那么就不应该指望死权重。

这个readme由大体上deepseek生成，我承认我懒得写。

---

**让 Mori 记住每一次对话，成为更懂你的 AI。**
