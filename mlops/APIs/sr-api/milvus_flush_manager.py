import time
import threading

FLUSH_INTERVAL = 5      # seconds
FLUSH_BATCH_LIMIT = 50  # embeddings
_last_flush_time = time.time()
_insert_count = 0
_lock = threading.Lock()

def register_insert(count=1):
    global _insert_count
    with _lock:
        _insert_count += count

def maybe_flush(collection):
    global _last_flush_time, _insert_count

    with _lock:
        now = time.time()

        if (
            _insert_count >= FLUSH_BATCH_LIMIT
            or now - _last_flush_time >= FLUSH_INTERVAL
        ):
            try:
                collection.flush()
                _insert_count = 0
                _last_flush_time = now
            except Exception:
                # Never crash request path due to flush
                pass
