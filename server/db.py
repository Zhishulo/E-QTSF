# -*- coding: utf-8 -*-
"""SQLite 访问层：线程安全连接 + 行字典化查询。"""
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "data", "output", "qms.db")

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


def query(sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
    rows = get_conn().execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def query_one(sql: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
    r = get_conn().execute(sql, params).fetchone()
    return dict(r) if r else None


def execute(sql: str, params: Tuple = ()) -> int:
    cur = get_conn().execute(sql, params)
    get_conn().commit()
    return cur.lastrowid


def init_placeholders(default_hash: str):
    """生成器写入的占位密码 → 默认口令 123456 的 PBKDF2 哈希。"""
    conn = get_conn()
    conn.execute(
        "UPDATE sys_user SET password_hash=? WHERE password_hash='PBKDF2$pending'",
        (default_hash,))
    conn.commit()
