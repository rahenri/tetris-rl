import logging
import time

def log_duration(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info("Function %s started", func.__qualname__)
        ret = func(*args, **kwargs)
        logging.info("Function %s took %fs", func.__qualname__, time.time() - start)
        return ret

    return wrapper
