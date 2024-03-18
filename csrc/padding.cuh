#pragma once
#include "helper.cuh"

template <typename Gemm>
class TestbedBatched_P : public BaseTestbed<Gemm>
{
public:
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementC = typename Gemm::ElementC;

    TestbedBatched_P(
        Options &options_,
        cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
        cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
        cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
        uint32_t seed_ = 3080) : BaseTestbed<Gemm>(options_, init_A_, init_B_, init_C_, seed_) {}

    /// Executes a batched kernel with padding and measures runtime
    Result profile()
    {
        std::cout << "Batched GEMM with Padding:\n"
                  << "====================================================" << std::endl;

        Result result;
        result.passed = false;

        // Initialize the problem
        this->allocate();
        this->initialize();

        int32_t padded_m = 0, padded_n = 0, padded_k = 0;
        for (auto problem : this->options.problem_sizes)
        {
            padded_m = std::max(padded_m, problem.m());
            padded_n = std::max(padded_n, problem.n());
            padded_k = std::max(padded_k, problem.k());
        }

        cutlass::DeviceAllocation<ElementA> block_A_padded;
        cutlass::DeviceAllocation<ElementB> block_B_padded;
        cutlass::DeviceAllocation<ElementC> block_C_padded;
        cutlass::DeviceAllocation<ElementC> block_D_padded;

        block_A_padded.reset(padded_m * padded_k * this->problem_count());
        block_B_padded.reset(padded_k * padded_n * this->problem_count());
        block_C_padded.reset(padded_m * padded_n * this->problem_count());
        block_D_padded.reset(padded_m * padded_n * this->problem_count());

        for (int i = 0; i < this->problem_count(); ++i)
        {
            auto const &problem = this->options.problem_sizes.at(i);
            int m = problem.m(), n = problem.n(), k = problem.k();

            cutlass::device_memory::copy_to_device(block_A_padded.get() + i * padded_m * padded_k,
                                                   this->block_A.get() + this->offset_A.at(i), m * k);

            cutlass::device_memory::copy_to_device(block_B_padded.get() + i * padded_k * padded_n,
                                                   this->block_B.get() + this->offset_B.at(i), k * n);

            cutlass::device_memory::copy_to_device(block_C_padded.get() + i * padded_m * padded_n,
                                                   this->block_C.get() + this->offset_C.at(i), m * n);
        }

        //
        // Prepare batched GEMM arguments
        //

        std::vector<ElementA const *> ptr_A_array;
        std::vector<ElementB const *> ptr_B_array;
        std::vector<ElementC const *> ptr_C_array;
        std::vector<ElementC *> ptr_D_array;

        for (int i = 0; i < this->problem_count(); ++i)
        {
            ptr_A_array.push_back(block_A_padded.get() + i * padded_m * padded_k);
            ptr_B_array.push_back(block_B_padded.get() + i * padded_k * padded_n);
            ptr_C_array.push_back(block_C_padded.get() + i * padded_m * padded_n);
            ptr_D_array.push_back(block_D_padded.get() + i * padded_m * padded_n);
        }

        // Copy argument arrays to device memory
        cutlass::DeviceAllocation<ElementA const *> ptr_A;
        cutlass::DeviceAllocation<ElementB const *> ptr_B;
        cutlass::DeviceAllocation<ElementC const *> ptr_C;
        cutlass::DeviceAllocation<ElementC *> ptr_D;

        ptr_A.reset(this->problem_count());
        ptr_B.reset(this->problem_count());
        ptr_C.reset(this->problem_count());
        ptr_D.reset(this->problem_count());

        ptr_A.copy_from_host(ptr_A_array.data());
        ptr_B.copy_from_host(ptr_B_array.data());
        ptr_C.copy_from_host(ptr_C_array.data());
        ptr_D.copy_from_host(ptr_D_array.data());

        typename Gemm::EpilogueOutputOp::Params epilogue_op(this->options.alpha, this->options.beta);

        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kArray,
            {padded_m, padded_n, padded_k},
            this->problem_count(),
            epilogue_op,
            (ElementA const **)ptr_A.get(),
            (ElementB const **)ptr_B.get(),
            (ElementC const **)ptr_C.get(),
            (ElementC **)ptr_D.get(),
            int64_t(),
            int64_t(),
            int64_t(),
            int64_t(),
            int64_t(padded_m),
            int64_t(padded_k),
            int64_t(padded_n),
            int64_t(padded_n)};

        Gemm gemm_op;
        cutlass::Status status = gemm_op.initialize(arguments);

        if (status != cutlass::Status::kSuccess)
        {
            std::cerr << "Failed to initialize CUTLASS Batched GEMM operator." << std::endl;
            return result;
        }

        // Warmup run
        status = gemm_op();

        if (status != cutlass::Status::kSuccess)
        {
            std::cerr << "CUTLASS GEMM operation failed." << std::endl;
            return result;
        }

        cudaEvent_t events[2];
        for (auto &event : events)
        {
            result.error = cudaEventCreate(&event);
            if (result.error != cudaSuccess)
            {
                std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
                return result;
            }
        }

        // Record an event at the start of a series of GEMM operations
        result.error = cudaEventRecord(events[0]);
        if (result.error != cudaSuccess)
        {
            std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        // Run profiling loop
        for (int iter = 0; iter < this->options.iterations; ++iter)
        {
            status = gemm_op();
            if (status != cutlass::Status::kSuccess)
            {
                std::cerr << "CUTLASS GEMM operation failed on iteration " << iter << std::endl;
                return result;
            }
        }

        // Record an event when the GEMM operations have been launched
        result.error = cudaEventRecord(events[1]);
        if (result.error != cudaSuccess)
        {
            std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        // Wait for work on the device to complete
        result.error = cudaEventSynchronize(events[1]);
        if (result.error != cudaSuccess)
        {
            std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        // Measure elapsed runtime
        float runtime_ms = 0;
        result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
        if (result.error != cudaSuccess)
        {
            std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        // Compute average runtime and GFLOPs
        result.runtime_ms = double(runtime_ms) / double(this->options.iterations);
        result.gflops = this->options.gflops(result.runtime_ms / 1000.0);

        // Cleanup
        for (auto event : events)
        {
            (void)cudaEventDestroy(event);
        }

        std::cout << std::endl;
        std::cout << "    Batched Runtime: " << result.runtime_ms << " ms" << std::endl;
        std::cout << "    Batched GFLOPs: " << result.gflops << std::endl;

        return result;
    }
};