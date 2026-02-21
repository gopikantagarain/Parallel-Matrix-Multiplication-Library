import numpy as np
from multiprocessing import Pool, cpu_count


def _compute_block(args):
    A, B, row_start, row_end, col_start, col_end = args
    block = np.dot(A[row_start:row_end, :], B[:, col_start:col_end])
    return (row_start, row_end, col_start, col_end, block)


def matmul_blockwise(A, B, block_size=256, processes=None):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions do not match.")

    m, k = A.shape
    k, n = B.shape

    processes = processes or cpu_count()
    C = np.zeros((m, n))

    tasks = []

    for row_start in range(0, m, block_size):
        row_end = min(row_start + block_size, m)

        for col_start in range(0, n, block_size):
            col_end = min(col_start + block_size, n)

            tasks.append((A, B, row_start, row_end, col_start, col_end))

    with Pool(processes) as pool:
        results = pool.map(_compute_block, tasks)

    for row_start, row_end, col_start, col_end, block in results:
        C[row_start:row_end, col_start:col_end] = block

    return C


def matmul_single(A, B):
    return np.dot(A, B)
