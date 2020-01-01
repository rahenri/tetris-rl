import logging
import time
import resource
from contextlib import contextmanager


def log_duration(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info("Function %s started", func.__qualname__)
        ret = func(*args, **kwargs)
        logging.info("Function %s took %fs", func.__qualname__, time.time() - start)
        return ret

    return wrapper


@contextmanager
def log_memory_change_context(name):
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    yield
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    change = after - before
    print(
        f"Block {name} changed memory by {change:.1f}MB "
        f"(before:{before:.1f}MB after:{after:.1f}MB)"
    )


def log_memory_change(func):
    def wrapper(*args, **kwargs):
        with log_memory_change_context(func.__qualname__):
            return func(*args, **kwargs)

    return wrapper
