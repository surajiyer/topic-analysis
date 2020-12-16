from typing import Union
import psutil
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    # joblib imports may raise DeprecationWarning on certain Python
    # versions
    import joblib
    from joblib import Parallel, delayed


__all__ = ["Parallel", "delayed", "chunker",
           "flatten", "preprocess_parallel"]


def chunker(iterable, total_length, chunksize):
    """Break an iterable into chunks"""
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))


def flatten(list_of_lists):
    """Flatten a list of lists to a combined list"""
    return [item for sublist in list_of_lists for item in sublist]


def preprocess_parallel(
        data, process_chunk: callable, n_jobs: int = 1, flatten : callable = flatten,
        chunksize: Union[int, str] = 'auto'):
    """Apply `process_chunk` on `data` parallely."""
    N = len(data)
    if chunksize == 'auto':
        cores = psutil.cpu_count(logical=False)
        if n_jobs > 0:
            cores = min(n_jobs, cores)
        chunksize = int(N / cores + 1)
    executor = Parallel(n_jobs=n_jobs, backend='loky', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(data, N, chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)
