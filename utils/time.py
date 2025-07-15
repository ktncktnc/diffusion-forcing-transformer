import time
from contextlib import contextmanager


@contextmanager
def timer(description="Code block"):
    """Context manager to time code execution"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"{description} took {elapsed:.4f} seconds")