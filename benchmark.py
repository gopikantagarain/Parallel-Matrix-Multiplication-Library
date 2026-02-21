import numpy as np
import time
import csv
from parallel_mm import matmul_blockwise, matmul_single
from multiprocessing import cpu_count


def run_benchmarks():
    sizes = [256, 512, 1024]
    processes = cpu_count()

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "single_time", "parallel_time", "speedup"])

        for size in sizes:
            print(f"Testing size {size}x{size}")

            A = np.random.rand(size, size)
            B = np.random.rand(size, size)

            start = time.perf_counter()
            C1 = matmul_single(A, B)
            single_time = time.perf_counter() - start

            start = time.perf_counter()
            C2 = matmul_blockwise(A, B)
            parallel_time = time.perf_counter() - start

            assert np.allclose(C1, C2, atol=1e-6)

            speedup = single_time / parallel_time
            writer.writerow([size, single_time, parallel_time, speedup])

    print("Benchmark saved to benchmark_results.csv")


if __name__ == "__main__":
    run_benchmarks()
