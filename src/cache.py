import sqlite3
import pickle
import time
from typing import Any, Optional

class Cache:
    """
    Small SQLite-backed key-value cache with TTL and pickling.
    Stores (key TEXT PRIMARY KEY, value BLOB, expires INTEGER)
    """
    def __init__(self, db_path: str = "./medico_cache.sqlite3"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        expires INTEGER
                    )""")
        self.conn.commit()

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        expires = int(time.time()) + ttl if ttl else None
        blob = pickle.dumps(value)
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO cache (key, value, expires) VALUES (?, ?, ?)", (key, blob, expires))
        self.conn.commit()

    def get(self, key: str):
        cur = self.conn.cursor()
        cur.execute("SELECT value, expires FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        blob, expires = row
        if expires and int(time.time()) > expires:
            # expired
            cur.execute("DELETE FROM cache WHERE key = ?", (key,))
            self.conn.commit()
            return None
        try:
            return pickle.loads(blob)
        except Exception:
            return None

    def delete(self, key: str):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM cache WHERE key = ?", (key,))
        self.conn.commit()
