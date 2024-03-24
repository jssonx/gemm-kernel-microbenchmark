import cutlass
import torch

dtype = torch.float16
plan = cutlass.op.GroupedGemm(element=dtype, layout=cutlass.LayoutType.RowMajor)

import random
random.seed(2023)

# Utility function to initialize A, B, C, and D matrices corresponding to dimensions M, N, and K
def initialize(dtype, M, N, K):
    sizes = [(M, K), (K, N), (M, N), (M, N)]
    return [torch.randint(-3, 3, size, device='cuda').to(dtype) for size in sizes]

# Utility function to generate `problems` GEMMs of random sizes
def generate_problems(problems):
    valid_sizes = [128, 256, 512, 1024]
    As, Bs, Cs, Ds = [], [], [], []
    for _ in range(problems):
        M, N, K = [random.choice(valid_sizes) for _ in range(3)]
        A, B, C, D = initialize(dtype, M, N, K)
        As.append(A)
        Bs.append(B)
        Cs.append(C)
        Ds.append(D)
    return As, Bs, Cs, Ds

As, Bs, Cs, Ds, = generate_problems(20)

import grouped_gemm

num_warmup = 20
num_profile = 100

# Warmup iterations
for _ in range(num_warmup):
    Ds = grouped_gemm.run(As, Bs)
    Ds_torch = [a @ b for a, b in zip(As, Bs)]
    torch.cuda.synchronize()

# Timing iterations
import time
grouped = 0
nongrouped = 0
for _ in range(num_profile):
    start = time.time()
    Ds = grouped_gemm.run(As, Bs)
    torch.cuda.synchronize()
    grouped += time.time() - start

    start = time.time()
    Ds_torch = [a @ b for a, b in zip(As, Bs)]
    torch.cuda.synchronize()
    nongrouped += time.time() - start

print('Grouped:     {:.3f} us'.format(grouped * 1e6/num_profile))
print('Non-Grouped: {:.3f} us'.format(nongrouped * 1e6/num_profile))
print('Speedup: {:.3f}'.format(nongrouped / grouped))