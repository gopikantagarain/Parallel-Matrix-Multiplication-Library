# Parallel Matrix Multiplication

## Overview

This project implements a **block-wise parallel matrix multiplication** library in Python using the `multiprocessing` module.
The objective is to improve the performance of large matrix multiplication by distributing computation across multiple CPU cores.

The project also includes performance benchmarking and edge-case testing to ensure correctness and efficiency.

---

## Features

* Block-wise matrix multiplication
* Parallel execution using Python multiprocessing
* Automatic utilization of multiple CPU cores
* Handles matrices not divisible by block size
* Performance comparison (single-process vs parallel)
* Benchmark results saved in CSV format
* Edge-case testing for robustness

---

## Project Structure

```
parallel-matrix-multiplication/
│
├── parallel_mm.py          # Core parallel matrix multiplication library
├── benchmark.py            # Performance benchmarking script
├── test_edge.py            # Edge case testing
├── benchmark_results.csv   # Generated benchmark results
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## Requirements

* Python 3.x
* NumPy

Install dependency:

```bash
pip install numpy
```

---

## Usage

### Parallel Matrix Multiplication

```python
import numpy as np
from parallel_mm import matmul_blockwise

A = np.random.rand(1000, 800)
B = np.random.rand(800, 600)

C = matmul_blockwise(A, B, block_size=256)
print(C.shape)
```

---

## Run Benchmark

To compare single-process and parallel performance:

```bash
python benchmark.py
```

This generates:

```
benchmark_results.csv
```

The file contains:

* Matrix size
* Single-process execution time
* Parallel execution time
* Speedup factor

---

## Run Edge Case Tests

```bash
python test_edge.py
```

Tests include:

* Non-divisible matrix sizes
* Small matrices
* Rectangular matrices
* Large block size handling
* Dimension mismatch error handling

---

## Performance

Parallel execution improves performance for large matrices by utilizing multiple CPU cores.

Typical speedup: **1.5x – 3x** depending on system configuration and matrix size.

---

## Concepts Used

* Parallel Computing
* Python Multiprocessing
* Block-wise computation
* Performance Benchmarking
* NumPy numerical operations

---

## Learning Outcomes

* Implemented CPU parallelization
* Compared single vs multi-process performance
* Handled real-world edge cases
* Built modular and testable Python code

---

## Future Improvements

* Shared memory optimization
* Automatic block size tuning
* GPU acceleration using CuPy
* Performance visualization with graphs

---

## Author

Gopikanta Garain

---

## Resume Summary

Developed a parallel matrix multiplication library using Python multiprocessing, achieving significant speedup over single-process execution with benchmarking and robust edge-case handling.
