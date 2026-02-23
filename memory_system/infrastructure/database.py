# database.py
import sqlite3
import json
from typing import Optional, Any
from ..config import Config
from ..models import MemoryItem
from ..utils import vector_to_blob, blob_to_vector


class Database:
    def __init__(self, config: Config):
        self.config = config
        self.conn = None
        self.cursor = None
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构"""
        # 首先创建数据库连接
        self.conn = sqlite3.connect(self.config.DB_PATH, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        
        # 创建游标
        self.cursor = self.conn.cursor()
        
        # 创建表
        # 记忆表
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.MEMORY_TABLE} (
                id TEXT PRIMARY KEY,
                vector BLOB,
                user_input TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                summary TEXT,
                heat INTEGER DEFAULT 0,
                created_turn INTEGER DEFAULT 0,
                last_interaction_turn INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 1,
                is_hot INTEGER DEFAULT 1,
                is_sleeping INTEGER DEFAULT 0,
                cluster_id TEXT,
                metadata TEXT,
                version INTEGER DEFAULT 1,
                update_count INTEGER DEFAULT 0,
                parent_turn INTEGER DEFAULT NULL
            )
        """)

        # 簇表
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.CLUSTER_TABLE} (
                id TEXT PRIMARY KEY,
                centroid BLOB,
                total_heat INTEGER DEFAULT 0,
                hot_memory_count INTEGER DEFAULT 0,
                cold_memory_count INTEGER DEFAULT 0,
                is_loaded INTEGER DEFAULT 0,
                size INTEGER DEFAULT 0,
                last_updated_turn INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                pending_heat_delta INTEGER DEFAULT 0,
                memory_additions_since_last_update INTEGER DEFAULT 0
            )
        """)

        # 热力池表
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.HEAT_POOL_TABLE} (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                heat_pool INTEGER DEFAULT {self.config.INITIAL_HEAT_POOL},
                unallocated_heat INTEGER DEFAULT {self.config.TOTAL_HEAT - self.config.INITIAL_HEAT_POOL},
                total_allocated_heat INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                current_turn INTEGER DEFAULT {self.config.INITIAL_TURN},
                pending_heat_json TEXT DEFAULT '{{}}'
            )
        """)

        # 操作日志表
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.OPERATION_LOG_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                operation_type TEXT,
                memory_id TEXT,
                cluster_id TEXT,
                old_value TEXT,
                new_value TEXT,
                turn INTEGER DEFAULT 0,
                applied INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0
            )
        """)
        
        # 联想图（Waypoint）表
        # self.cursor.execute(f"""
        #     CREATE TABLE IF NOT EXISTS {self.config.WAYPOINT_TABLE} (
        #         source_id TEXT,
        #         target_id TEXT,
        #         weight REAL DEFAULT 0.5,
        #         created_turn INTEGER DEFAULT 0,
        #         last_updated_turn INTEGER DEFAULT 0,
        #         PRIMARY KEY (source_id, target_id)
        #     )
        # """)
        # self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_waypoint_source ON {self.config.WAYPOINT_TABLE}(source_id)")

        # 新增：暂存热力单元表（替代旧的 pending_heat 表）
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.PENDING_HEAT_UNITS_TABLE} (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                pending_heat INTEGER NOT NULL,
                created_turn INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                version INTEGER DEFAULT 1
            )
        """)
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_pending_units_status 
            ON {self.config.PENDING_HEAT_UNITS_TABLE}(status)
        """)

        # 索引
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_op_log_turn ON {self.config.OPERATION_LOG_TABLE}(turn)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_op_log_applied ON {self.config.OPERATION_LOG_TABLE}(applied)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_cluster ON {self.config.MEMORY_TABLE}(cluster_id, heat)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_hot_heat ON {self.config.MEMORY_TABLE}(is_hot, heat DESC)")
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_memory_turn ON {self.config.MEMORY_TABLE}(last_interaction_turn DESC)")

        # 初始化热力池记录
        self.cursor.execute(f"""
            INSERT OR IGNORE INTO {self.config.HEAT_POOL_TABLE} 
            (id, heat_pool, unallocated_heat, total_allocated_heat, current_turn)
            VALUES (1, {self.config.INITIAL_HEAT_POOL}, 
                   {self.config.TOTAL_HEAT - self.config.INITIAL_HEAT_POOL}, 0, {self.config.INITIAL_TURN})
        """)

        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    def execute(self, sql, params=()):
        return self.cursor.execute(sql, params)

    def executemany(self, sql, params):
        return self.cursor.executemany(sql, params)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    # 常用查询封装
    def load_hot_memories(self, limit=1000):
        self.cursor.execute(f"""
            SELECT * FROM {self.config.MEMORY_TABLE} 
            WHERE is_hot = 1
            ORDER BY heat DESC
            LIMIT ?
        """, (limit,))
        rows = self.cursor.fetchall()
        memories = []
        for row in rows:
            memory = MemoryItem(
                id=row['id'],
                vector=blob_to_vector(row['vector']),
                user_input=row['user_input'],
                ai_response=row['ai_response'],
                summary=row['summary'] or "",
                heat=row['heat'],
                created_turn=row['created_turn'],
                last_interaction_turn=row['last_interaction_turn'],
                access_count=row['access_count'],
                is_hot=bool(row['is_hot']),
                is_sleeping=bool(row['is_sleeping']),
                cluster_id=row['cluster_id'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                version=row['version'],
                update_count=row['update_count'] or 0,
                parent_turn=row['parent_turn']
            )
            memories.append(memory)
        return memories

    def load_all_clusters(self):
        self.cursor.execute(f"SELECT * FROM {self.config.CLUSTER_TABLE}")
        rows = self.cursor.fetchall()
        clusters = []
        for row in rows:
            from ..models import SemanticCluster
            cluster = SemanticCluster(
                id=row['id'],
                centroid=blob_to_vector(row['centroid']),
                total_heat=row['total_heat'],
                hot_memory_count=row['hot_memory_count'],
                cold_memory_count=row['cold_memory_count'],
                is_loaded=bool(row['is_loaded']),
                size=row['size'],
                last_updated_turn=row['last_updated_turn'],
                version=row['version'],
                pending_heat_delta=row['pending_heat_delta'],
                memory_additions_since_last_update=row['memory_additions_since_last_update'] or 0
            )
            clusters.append(cluster)
        return clusters

    def load_heat_pool_state(self):
        self.cursor.execute(
            f"SELECT heat_pool, unallocated_heat, total_allocated_heat, current_turn "
            f"FROM {self.config.HEAT_POOL_TABLE} WHERE id = 1"
        )
        return self.cursor.fetchone()