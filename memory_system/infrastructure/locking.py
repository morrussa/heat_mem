import threading
import time
from typing import Dict


class DistributedLockManager:
    def __init__(self):
        self.locks: Dict[str, threading.RLock] = {}
        self.lock_timestamps: Dict[str, float] = {}
        self.lock_threads: Dict[str, int] = {}
        self.global_lock = threading.RLock()

    def acquire(self, lock_key: str, timeout: float = 5.0) -> bool:
        start_time = time.time()
        thread_id = threading.get_ident()

        with self.global_lock:
            if lock_key in self.lock_threads and self.lock_threads[lock_key] == thread_id:
                return True

            while time.time() - start_time < timeout:
                if lock_key not in self.locks:
                    self.locks[lock_key] = threading.RLock()
                    self.lock_timestamps[lock_key] = time.time()
                    self.lock_threads[lock_key] = thread_id
                    return True
                elif self.lock_timestamps[lock_key] + timeout < time.time():
                    del self.locks[lock_key]
                    del self.lock_timestamps[lock_key]
                    if lock_key in self.lock_threads:
                        del self.lock_threads[lock_key]
                    continue

                time.sleep(0.01)

        return False

    def release(self, lock_key: str):
        with self.global_lock:
            thread_id = threading.get_ident()
            if lock_key in self.lock_threads and self.lock_threads[lock_key] == thread_id:
                if lock_key in self.locks:
                    del self.locks[lock_key]
                    del self.lock_timestamps[lock_key]
                del self.lock_threads[lock_key]

    def with_lock(self, lock_key: str, timeout: float = 5.0):
        class LockContext:
            def __init__(self, manager, lock_key, timeout):
                self.manager = manager
                self.lock_key = lock_key
                self.timeout = timeout
                self.acquired = False

            def __enter__(self):
                self.acquired = self.manager.acquire(self.lock_key, self.timeout)
                if not self.acquired:
                    raise TimeoutError(f"Failed to acquire lock {self.lock_key} within {self.timeout} seconds")
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.acquired:
                    self.manager.release(self.lock_key)

        return LockContext(self, lock_key, timeout)