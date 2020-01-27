import queue
import threading


class ConfigManager:
    def __init__(self, setter, initial_config):
        self._current_config = initial_config
        self._setter = setter
        self._queue = queue.Queue(1)
        self._config_lock = threading.Lock()

    def push_update(self, new_config):
        resp_queue = queue.Queue(1)
        self._queue.put((resp_queue, new_config))
        resp = resp_queue.get()
        return resp

    def current_config(self):
        with self._config_lock:
            return self._current_config.copy()

    def handle_events(self):
        try:
            resp_queue, new_config = self._queue.get_nowait()
        except queue.Empty:
            return
        updated, message = self._setter(new_config)
        if updated:
            with self._config_lock:
                self._current_config = new_config
        resp_queue.put((updated, message))
