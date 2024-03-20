# moe

```bash
~/pg/moe$ ./csrc/bench/main --groups=16 --n=64 --k=768 --iterations=3
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


```bash
~/pg/moe$ ./csrc/bench/main --groups=16 --n=3072 --k=768 --iterations=3
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

```bash
./csrc/bench/main --groups=16 --n=64 --k=12288 --iterations=1
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