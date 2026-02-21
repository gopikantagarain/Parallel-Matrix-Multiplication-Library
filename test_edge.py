import numpy as np
from parallel_mm import matmul_blockwise, matmul_single


def test_non_divisible():
    print("Test 1: Non-divisible dimensions")
    A = np.random.rand(530, 410)
    B = np.random.rand(410, 370)
    C1 = matmul_single(A, B)
    C2 = matmul_blockwise(A, B, block_size=128)
    print("Passed:", np.allclose(C1, C2))


def test_small_matrix():
    print("\nTest 2: Very small matrices")
    A = np.random.rand(2, 3)
    B = np.random.rand(3, 4)
    C1 = matmul_single(A, B)
    C2 = matmul_blockwise(A, B, block_size=10)
    print("Passed:", np.allclose(C1, C2))


def test_rectangular_matrix():
    print("\nTest 3: Rectangular matrices")
    A = np.random.rand(300, 50)
    B = np.random.rand(50, 700)
    C1 = matmul_single(A, B)
    C2 = matmul_blockwise(A, B, block_size=64)
    print("Passed:", np.allclose(C1, C2))


def test_large_block_size():
    print("\nTest 4: Block size larger than matrix")
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    C1 = matmul_single(A, B)
    C2 = matmul_blockwise(A, B, block_size=500)
    print("Passed:", np.allclose(C1, C2))


def test_dimension_mismatch():
    print("\nTest 5: Dimension mismatch (should raise error)")
    A = np.random.rand(100, 50)
    B = np.random.rand(60, 100)
    try:
        matmul_blockwise(A, B)
        print("Failed: No error raised")
    except ValueError:
        print("Passed: Error raised correctly")


def main():
    test_non_divisible()
    test_small_matrix()
    test_rectangular_matrix()
    test_large_block_size()
    test_dimension_mismatch()


if __name__ == "__main__":
    main()
