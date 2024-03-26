# GEMM Kernel Microbenchmark

This repo provides a microbenchmark for GEMM kernels on NVIDIA GPUs with Ampere Architecture (sm_80). It includes both a CUDA kernel benchmark and a Python extension benchmark.

## Requirements
- NVIDIA GPU with Ampere Architecture (sm_80)
- CUDA 12.2

## Getting Started

### CUDA Kernel Benchmark

1. Build the project:
```bash
$ make
```

2. Run a benchmark with specific parameters:
```bash
$ ./csrc/bench/main --groups=16 --m=64 --n=64 --k=768 --iterations=3
```
Where:
- --groups: Number of groups
- --m, --n, --k: Problem size dimensions
- --iterations: Number of iterations

3. For more information on available options:
```bash
$ ./csrc/bench/main --help
```

### Python Extension Benchmark

1. Export the CUDA kernel as a Python extension:
```bash
$ python ./python/testbed/lib.py
$ cd out && TORCH_CUDA_ARCH_LIST="8.0" python setup.py install --user
```

2. Run the benchmark:
```bash
$ python ./python/testbed/multi_gemm.py > perf.txt
```

## References

- CUTLASS Examples
  - ["02_pytorch_extension_grouped_gemm" Notebook](https://github.com/NVIDIA/cutlass/blob/main/examples/python/02_pytorch_extension_grouped_gemm.ipynb): A guide to implementing grouped GEMM operations as PyTorch extensions.
  - ["gemm_grouped" CUDA Example](https://github.com/NVIDIA/cutlass/blob/main/examples/24_gemm_grouped/gemm_grouped.cu): Example code and documentation for grouped GEMM operations in CUDA.