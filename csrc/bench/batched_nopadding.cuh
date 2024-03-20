#pragma once
#include "helper.cuh"

template <typename Gemm>
class TestbedBatched_NP : public BaseTestbed<Gemm>
{
public:
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementC = typename Gemm::ElementC;

    TestbedBatched_NP(
        Options &options_,
        cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
        cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
        cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
        uint32_t seed_ = 3080) : BaseTestbed<Gemm>(options_, init_A_, init_B_, init_C_, seed_) {}

    /// Executes a batched kernel and measures runtime
    Result profile()
    {
        std::cout << "\nBatched GEMM without padding:\n"
                  << "====================================================" << std::endl;

        Result result;
        result.passed = false;

        // Initialize the problem
        this->allocate();
        this->initialize();

        int32_t lda = this->lda_host[0];
        int32_t ldb = this->ldb_host[0];
        int32_t ldc = this->ldc_host[0];

        //
        // Prepare batched GEMM arguments
        //

        std::vector<ElementA const *> ptr_A_array;
        std::vector<ElementB const *> ptr_B_array;
        std::vector<ElementC const *> ptr_C_array;
        std::vector<ElementC *> ptr_D_array;

        for (int i = 0; i < this->problem_count(); ++i)
        {
            ptr_A_array.push_back(this->block_A.get() + this->offset_A.at(i));
            ptr_B_array.push_back(this->block_B.get() + this->offset_B.at(i));
            ptr_C_array.push_back(this->block_C.get() + this->offset_C.at(i));
            ptr_D_array.push_back(this->block_D.get() + this->offset_D.at(i));
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
            this->options.problem_sizes[0],
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
            int64_t(lda),
            int64_t(ldb),
            int64_t(ldc),
            int64_t(ldc)};

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

        // Verify results
        if (this->options.reference_check)
        {
            result.passed = this->verify();
        }
        else
        {
            result.passed = true;
        }

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