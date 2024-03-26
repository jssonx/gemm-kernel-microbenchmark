# Results


# Summary

- --groups=16 --n=64 --k=768 --iterations=3
    - **Grouped GEMM** > Batched GEMM with Padding > Batched GEMM with bins

- --groups=16 --n=3072 --k=768 --iterations=3
    - **Grouped GEMM** > Batched GEMM with bins > Batched GEMM with Padding

- --groups=16 --n=64 --k=12288 --iterations=3
    - **Grouped GEMM** > Batched GEMM with Padding > Batched GEMM with bins

- --groups=16 --m=64 --n=64 --k=768 --iterations=3
    - Batched GEMM with Padding > **Grouped GEMM** > Batched GEMM with bins

## Inbalanced Case

### ./csrc/bench/main --groups=16 --n=64 --k=768 --iterations=3
Grouped GEMM > Batched GEMM with Padding > Batched GEMM with bins
```bash
Batched GEMM with bins:
====================================================
    15 batched GEMMs launched

    Batched Runtime: 358.632 ms
    Batched  GFLOPs: 20.841

Batched GEMM without padding:
====================================================

***
Error - problem 1 failed the QA check
***


    Batched Runtime: 52.5254 ms
    Batched GFLOPs: 142.298

Batched GEMM with Padding:
====================================================

    Batched Runtime: 177.511 ms
    Batched GFLOPs: 42.1059

Grouped GEMM (CUTLASS) with mode kDeviceOnly:
====================================================
    594 total threadblock tiles.

    Grouped Runtime: 95.1023 ms
    Grouped  GFLOPs: 78.5917

Passed
```

### ./csrc/bench/main --groups=16 --n=3072 --k=768 --iterations=3
Grouped GEMM > Batched GEMM with bins > Batched GEMM with Padding
```bash
Batched GEMM with bins:
====================================================
    15 batched GEMMs launched

    Batched Runtime: 2126.48 ms
    Batched  GFLOPs: 168.712

Batched GEMM without padding:
====================================================

***
Error - problem 1 failed the QA check
***


    Batched Runtime: 751.831 ms
    Batched GFLOPs: 477.187

Batched GEMM with Padding:
====================================================

    Batched Runtime: 4172.05 ms
    Batched GFLOPs: 85.9923

Grouped GEMM (CUTLASS) with mode kDeviceOnly:
====================================================
    14256 total threadblock tiles.

    Grouped Runtime: 2145.59 ms
    Grouped  GFLOPs: 167.21

Passed
```

### ./csrc/bench/main --groups=16 --n=64 --k=12288 --iterations=3
Grouped GEMM > Batched GEMM with Padding > Batched GEMM with bins
```bash
Batched GEMM with bins:
====================================================
    15 batched GEMMs launched

    Batched Runtime: 4928.01 ms
    Batched  GFLOPs: 24.267

Batched GEMM without padding:
====================================================

***
Error - problem 1 failed the QA check
***


    Batched Runtime: 733.139 ms
    Batched GFLOPs: 163.118

Batched GEMM with Padding:
====================================================

    Batched Runtime: 2508.34 ms
    Batched GFLOPs: 47.6762

Grouped GEMM (CUTLASS) with mode kDeviceOnly:
====================================================
    594 total threadblock tiles.

    Grouped Runtime: 1334.33 ms
    Grouped  GFLOPs: 89.624

Passed
```

## Balanced Case

### ./csrc/bench/main --groups=16 --m=64 --n=64 --k=768 --iterations=3
Batched GEMM with Padding > Grouped GEMM > Batched GEMM with bins
```bash
Batched GEMM with bins:
====================================================
    1 batched GEMMs launched

    Batched Runtime: 28.8177 ms
    Batched  GFLOPs: 3.4931

Batched GEMM without padding:
====================================================

    Batched Runtime: 25.6044 ms
    Batched GFLOPs: 3.93148

Batched GEMM with Padding:
====================================================

    Batched Runtime: 23.5527 ms
    Batched GFLOPs: 4.27396

Grouped GEMM (CUTLASS) with mode kDeviceOnly:
====================================================
    16 total threadblock tiles.

    Grouped Runtime: 25.8635 ms
    Grouped  GFLOPs: 3.8921

Passed
```