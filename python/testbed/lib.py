import cutlass
import torch
import sys

plan = cutlass.op.GroupedGemm(element=torch.float16, layout=cutlass.LayoutType.RowMajor)
op = plan.construct()
cutlass.emit.pytorch(op, name='grouped_gemm', cc=plan.cc, sourcedir='out', jit=False)

# cd out && TORCH_CUDA_ARCH_LIST="8.0" python setup.py install --user
# sys.path.append("~/.local/lib/python3.11/site-packages")