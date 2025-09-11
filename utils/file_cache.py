import json
import os
import time
import hashlib
from typing import Any


class FileCache:
    def __init__(self, root: str, ttl_days: int = 30):
        self.root = root
        self.ttl = ttl_days * 86400
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.root, f"{h}.json")

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        if self.ttl > 0 and (time.time() - os.path.getmtime(p)) > self.ttl:
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, default=str)
