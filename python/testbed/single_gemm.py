import torch
import time
import sys
sys.path.append("~/.local/lib/python3.11/site-packages")
import grouped_gemm

class TestModule(torch.nn.Module):
    def __init__(self, hidden_size, ffn_dim):
        super(TestModule, self).__init__()
        self.wi = torch.nn.Linear(hidden_size, ffn_dim, bias=False)

    def forward(self, hidden_states, iterations=100):
        if len(hidden_states.shape) == 2 and hidden_states.shape[0] > 0:
            # Warm up
            for _ in range(10):
                A1 = [hidden_states]
                B1 = [self.wi.weight.data.t()]
                C1 = grouped_gemm.run(A1, B1)
                C2 = self.wi(hidden_states)

            # grouped_gemm
            grouped_start_time = time.time()
            for _ in range(iterations):
                A1 = [hidden_states]
                B1 = [self.wi.weight.data.t()]
                D1 = grouped_gemm.run(A1, B1)
            torch.cuda.synchronize()
            grouped_end_time = time.time()
            grouped_avg_time = (grouped_end_time - grouped_start_time) / iterations

            # nn.Linear
            linear_start_time = time.time()
            for _ in range(iterations):
                h3 = self.wi(hidden_states)
            torch.cuda.synchronize()
            linear_end_time = time.time()
            linear_avg_time = (linear_end_time - linear_start_time) / iterations

            print(f"Average grouped time over {iterations} iterations: {grouped_avg_time:e}")
            print(f"Average linear time over {iterations} iterations: {linear_avg_time:e}")

            return D1

def benchmark():
    batch_size = 32
    sequence_length = 512
    hidden_size = 768
    ffn_dim = 3072

    # create a random hidden_states tensor on the GPU
    hidden_states = torch.randn(batch_size * sequence_length, hidden_size, device='cuda', dtype=torch.float16)

    # create the test module
    test_module = TestModule(hidden_size, ffn_dim).to('cuda').half()

    # run the benchmark
    test_module(hidden_states)

benchmark()