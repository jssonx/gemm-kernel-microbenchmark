import torch
import time
import sys
sys.path.append("~/.local/lib/python3.11/site-packages")
import grouped_gemm
import random

def stack_or_pad(hidden_states_list):
    if all(hidden_state.shape == hidden_states_list[0].shape for hidden_state in hidden_states_list):
        return torch.stack(hidden_states_list, dim=0)
    else:
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
            for _ in range(20):
                C1 = grouped_gemm.run(hidden_states_list, weights)
                C2 = [self.wi(hidden_states) for hidden_states in hidden_states_list]
                C3 = [a @ b for a, b in zip(hidden_states_list, weights)]
                C4 = torch.bmm(stack_or_pad(hidden_states_list), torch.stack(weights, dim=0))

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
            stack_states = stack_or_pad(hidden_states_list)
            stack_weights = torch.stack(weights, dim=0)
            bmm_start_time = time.time()
            for _ in range(iterations):
                C4 = torch.bmm(stack_states, stack_weights)
                torch.cuda.synchronize()
            bmm_end_time = time.time()
            bmm_avg_time = (bmm_end_time - bmm_start_time) / iterations
            
            total_flops = sum(2 * h.shape[0] * h.shape[1] * w.shape[1] for h, w in zip(hidden_states_list, weights))
            
            return total_flops, grouped_avg_time, linear_avg_time, a_b_avg_time, bmm_avg_time

def benchmark(batch_size, sequence_length, hidden_size, ffn_dim):
    num_hidden_states = 8
    iterations = 100

    hidden_states_list = [torch.randn(random.randint(1, batch_size * sequence_length), hidden_size, device='cuda', dtype=torch.float16) for _ in range(num_hidden_states)]    
    initial_weight = torch.randn(hidden_size, ffn_dim, device='cuda', dtype=torch.float16)
    weights = [initial_weight.clone() for _ in range(num_hidden_states)]

    test_module = TestModule(hidden_size, ffn_dim).to('cuda').half()
    test_module.wi.weight.data = initial_weight.t()
    total_flops, grouped_avg_time, linear_avg_time, a_b_avg_time, bmm_avg_time = test_module(hidden_states_list, weights, iterations)
    return total_flops, grouped_avg_time, linear_avg_time, a_b_avg_time, bmm_avg_time

def search_space():

    batch_sizes = [1]
    sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]
    hidden_sizes = [128, 768, 4096]
    ffn_dims = [256, 512, 1024, 2048, 4096, 8192, 14336]
    iterations = 10
    
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            for hidden_size in hidden_sizes:
                for ffn_dim in ffn_dims:
                    print(f"Parameters: batch_size={batch_size}, sequence_length={sequence_length}, hidden_size={hidden_size}, ffn_dim={ffn_dim}")
                    total_flops_list = []
                    grouped_avg_time_list = []
                    linear_avg_time_list = []
                    a_b_avg_time_list = []
                    bmm_avg_time_list = []
                    
                    for _ in range(iterations):
                        total_flops, grouped_avg_time, linear_avg_time, a_b_avg_time, bmm_avg_time = benchmark(batch_size, sequence_length, hidden_size, ffn_dim)
                        total_flops_list.append(total_flops)
                        grouped_avg_time_list.append(grouped_avg_time)
                        linear_avg_time_list.append(linear_avg_time)
                        a_b_avg_time_list.append(a_b_avg_time)
                        bmm_avg_time_list.append(bmm_avg_time)
                    
                    tflops_grouped_gemm = sum(total_flops_list) / (sum(grouped_avg_time_list) * 1e12)
                    tflops_linear = sum(total_flops_list) / (sum(linear_avg_time_list) * 1e12)
                    tflops_a_b = sum(total_flops_list) / (sum(a_b_avg_time_list) * 1e12)
                    tflops_bmm = sum(total_flops_list) / (sum(bmm_avg_time_list) * 1e12)
                    
                    print(f"TFLOPS (grouped_gemm): {tflops_grouped_gemm:.3f}")
                    print(f"TFLOPS (nn.Linear): {tflops_linear:.3f}")
                    print(f"TFLOPS (a @ b): {tflops_a_b:.3f}")
                    print(f"TFLOPS (bmm): {tflops_bmm:.3f}")
                    
                    print("\n" + "-" * 50 + "\n")

search_space()

# batch_sizes = [1]
# sequence_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# hidden_sizes = [128, 768, 4096]
# ffn_dims = [256, 512, 1024, 2048, 4096, 8192, 14336]

# batch_sizes = [2]
# sequence_lengths = [32768]
# hidden_sizes = [4096]
# ffn_dims = [14336]

# batch_sizes = [1]
# sequence_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]
# hidden_sizes = [4096]
# ffn_dims = [14336]