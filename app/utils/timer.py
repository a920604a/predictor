import functools
import time


def time_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func_return_val = func(*args, **kwargs)
        end = time.perf_counter()
        print('{0:<1}.{1:<8} : {2:<8}'.format('module', 'function', 'time'))
        return func_return_val
    return wrapper
