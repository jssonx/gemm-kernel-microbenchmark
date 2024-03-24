import torch
import time
import sys
sys.path.append("~/.local/lib/python3.11/site-packages")
import grouped_gemm
import random


def pad_and_stack(hidden_states_list):
    max_seq_length = max(hidden_state.shape[0] for hidden_state in hidden_states_list)
    hidden_size = hidden_states_list[0].shape[1]
    dtype = hidden_states_list[0].dtype
    device = hidden_states_list[0].device
    padded_tensor = torch.zeros(len(hidden_states_list), max_seq_length, hidden_size, dtype=dtype, device=device)
    for i, hidden_state in enumerate(hidden_states_list):
        seq_length = hidden_state.shape[0]
        padded_tensor[i, :seq_length, :] = hidden_state

    return padded_tensor

class TestModule(torch.nn.Module):
    def __init__(self, hidden_size, ffn_dim):
        super(TestModule, self).__init__()
        self.wi = torch.nn.Linear(hidden_size, ffn_dim, bias=False)

    def forward(self, hidden_states_list, weights, iterations=100):
        if len(hidden_states_list) > 0 and hidden_states_list[0].shape[0] > 0:
            
            # Warm up
            for _ in range(100):
                C1 = grouped_gemm.run(hidden_states_list, weights)
                C2 = [self.wi(hidden_states) for hidden_states in hidden_states_list]
                C3 = [a @ b for a, b in zip(hidden_states_list, weights)]
                C4 = torch.bmm(pad_and_stack(hidden_states_list), torch.stack(weights, dim=0))

            # grouped_gemm
            grouped_start_time = time.time()
            for _ in range(iterations):
                C1 = grouped_gemm.run(hidden_states_list, weights)
                torch.cuda.synchronize()
            grouped_end_time = time.time()
            grouped_avg_time = (grouped_end_time - grouped_start_time) / iterations

            # nn.Linear
            linear_start_time = time.time()
            for _ in range(iterations):
                C2 = [self.wi(hidden_states) for hidden_states in hidden_states_list]
                torch.cuda.synchronize()
            linear_end_time = time.time()
            linear_avg_time = (linear_end_time - linear_start_time) / iterations
            
            # a @ b
            a_b_start_time = time.time()
            for _ in range(iterations):
                C3 = [a @ b for a, b in zip(hidden_states_list, weights)]
                torch.cuda.synchronize()
            a_b_end_time = time.time()
            a_b_avg_time = (a_b_end_time - a_b_start_time) / iterations
            
            # bmm
            bmm_start_time = time.time()
            for _ in range(iterations):
                C4 = torch.bmm(pad_and_stack(hidden_states_list), torch.stack(weights, dim=0))
                torch.cuda.synchronize()
            bmm_end_time = time.time()
            bmm_avg_time = (bmm_end_time - bmm_start_time) / iterations
            
            print(f"Iterations: {iterations}")
            print("======================")
            print(f"Average grouped_gemm time: {grouped_avg_time * 1e6:.3f} μs")
            print(f"Average nn.Linear time: {linear_avg_time * 1e6:.3f} μs")
            print(f"Average a @ b time: {a_b_avg_time * 1e6:.3f} μs")
            print(f"Average bmm time: {bmm_avg_time * 1e6:.3f} μs")
            print("\n======================")
            print(f"Speedup (nn.Linear / grouped_gemm): {linear_avg_time / grouped_avg_time:.3f}")
            print(f"Speedup (a @ b / grouped_gemm): {a_b_avg_time / grouped_avg_time:.3f}")
            print(f"Speedup (bmm / grouped_gemm): {bmm_avg_time / grouped_avg_time:.3f}")

            return C1, C2, C3, C4

def benchmark():
    batch_size = 16
    sequence_length = 64
    hidden_size = 768
    ffn_dim = hidden_size * 4
    num_hidden_states = 8
    iterations = 300

    hidden_states_list = [torch.randn(random.randint(1, batch_size * sequence_length), hidden_size, device='cuda', dtype=torch.float16) for _ in range(num_hidden_states)]    
    initial_weight = torch.randn(hidden_size, ffn_dim, device='cuda', dtype=torch.float16)
    weights = [initial_weight.clone() for _ in range(num_hidden_states)]

    test_module = TestModule(hidden_size, ffn_dim).to('cuda').half()
    test_module.wi.weight.data = initial_weight.t()
    test_module(hidden_states_list, weights, iterations)

benchmark()