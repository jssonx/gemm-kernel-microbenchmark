# MoE Kernel

## Requirements
- NVIDIA GPU with Ampere Architecture (sm_80)
- CUDA 12.2

## Getting Started

Build the project:
```bash
$ make
```

To run a benchmark with 16 groups, 64x64x768 problem size, and 3 iterations:
```bash
$ ./csrc/bench/main --groups=16 --m=64 --n=64 --k=768 --iterations=3
```

For more information on available options:
```bash
$ ./csrc/bench/main --help
```