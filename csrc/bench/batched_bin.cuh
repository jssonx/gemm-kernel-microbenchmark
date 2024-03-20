#pragma once
#include "helper.cuh"

template <typename Gemm>
class TestbedBatched : BaseTestbed<Gemm>
{
public:
    TestbedBatched(
        Options &options_,
        cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
        cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
        cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
        uint32_t seed_ = 3080) : BaseTestbed<Gemm>(options_, init_A_, init_B_, init_C_, seed_) {}

    void print_problem_sizes()
    {
        std::cout << std::endl;
        size_t bin_idx = 0;
        size_t problem_count_check = 0;
        std::cout << "Conventionally executed as " << this->options.problem_bins.size() << " batched GEMMs:\n";
        for (auto const &bin : this->options.problem_bins)
        {

            std::cout << "  [" << bin_idx << "]: "
                      << bin.first.m() << "-by-" << bin.first.n() << "-by-" << bin.first.k()
                      << ", batch count: " << bin.second.size() << "\n";

            ++bin_idx;
            problem_count_check += bin.second.size();
        }

        if (problem_count_check != this->problem_count())
        {
            std::cout << "\n***\nERROR in BINNING LOGIC!\n***\n"
                      << std::endl;
        }

        std::cout << std::endl;
    }

    /// Executes a batched kernel and measures runtime
    Result profile()
    {
        std::cout << "Batched GEMM with bins:\n"
                  << "====================================================" << std::endl;

        Result result;
        result.passed = false;

        // Initialize the problem
        this->allocate();
        this->initialize();

        if (this->options.verbose)
        {
            print_problem_sizes();
        }

        //
        // Prepare batched GEMM environment
        //

        int32_t effective_streams = (this->options.cuda_streams ? this->options.cuda_streams : 1);

        // Array of leading dimensions used by batched GEMM calls
        std::vector<cutlass::gemm::GemmCoord> bin_problem_sizes;
        std::vector<int32_t> bin_count;
        std::vector<int32_t> bin_ldm_A;
        std::vector<int32_t> bin_ldm_B;
        std::vector<int32_t> bin_ldm_C;
        std::vector<int32_t> bin_start;

        std::vector<void const *> ptr_A_batched_host;
        std::vector<void const *> ptr_B_batched_host;
        std::vector<void *> ptr_C_batched_host;

        for (auto const &bin : this->options.problem_bins)
        {
            int first_idx = bin.second.front();

            bin_problem_sizes.push_back(this->options.problem_sizes.at(first_idx));
            bin_count.push_back(int32_t(bin.second.size()));

            bin_ldm_A.push_back(static_cast<int32_t>(this->lda_host.at(first_idx)));
            bin_ldm_B.push_back(static_cast<int32_t>(this->ldb_host.at(first_idx)));
            bin_ldm_C.push_back(static_cast<int32_t>(this->ldc_host.at(first_idx)));

            if (ptr_A_batched_host.size() % 2)
            {
                ptr_A_batched_host.push_back(nullptr);
                ptr_B_batched_host.push_back(nullptr);
                ptr_C_batched_host.push_back(nullptr);
            }

            bin_start.push_back(int32_t(ptr_A_batched_host.size()));

            for (int idx : bin.second)
            {

                if (bin_problem_sizes.back() != this->options.problem_sizes.at(idx))
                {
                    std::cerr << "Error - failed to group problems.\n";
                    return result;
                }

                if (bin_ldm_A.back() != this->lda_host.at(idx))
                {
                    std::cerr << "Error - failed to group problems.\n";
                    return result;
                }

                if (bin_ldm_B.back() != this->ldb_host.at(idx))
                {
                    std::cerr << "Error - failed to group problems.\n";
                    return result;
                }

                if (bin_ldm_C.back() != this->ldc_host.at(idx))
                {
                    std::cerr << "Error - failed to group problems.\n";
                    return result;
                }

                ptr_A_batched_host.push_back(this->block_A.get() + this->offset_A.at(idx));
                ptr_B_batched_host.push_back(this->block_B.get() + this->offset_B.at(idx));
                ptr_C_batched_host.push_back(this->block_D.get() + this->offset_C.at(idx));
            }
        }

        // Array of GMEM pointers used by batched array GEMM calls
        cutlass::DeviceAllocation<void const *> ptr_A_batched;
        cutlass::DeviceAllocation<void const *> ptr_B_batched;
        cutlass::DeviceAllocation<void *> ptr_C_batched;

        ptr_A_batched.reset(ptr_A_batched_host.size());
        ptr_B_batched.reset(ptr_A_batched_host.size());
        ptr_C_batched.reset(ptr_A_batched_host.size());

        ptr_A_batched.copy_from_host(ptr_A_batched_host.data());
        ptr_B_batched.copy_from_host(ptr_B_batched_host.data());
        ptr_C_batched.copy_from_host(ptr_C_batched_host.data());

        //
        // Create CUDA streams to maximize concurrency of batched-array GEMM kernels
        //
        std::vector<cudaStream_t> cuda_streams;

        //
        // Warmup run
        //

        if (this->options.cuda_streams)
        {
            for (int i = 0; i < this->options.cuda_streams; ++i)
            {
                cudaStream_t stream;

                result.error = cudaStreamCreate(&stream);
                if (result.error != cudaSuccess)
                {
                    std::cerr << "Failed to create CUDA stream." << std::endl;
                    return result;
                }
                cuda_streams.push_back(stream);
            }
        }
        else
        {
            cuda_streams.push_back(nullptr);
        }

        // Use 'D' for the in/out workspace
        this->block_D.copy_from_device(this->block_C.get());

        for (int bin_idx = 0; bin_idx < int32_t(bin_problem_sizes.size()); ++bin_idx)
        {

            cutlass::gemm::GemmCoord const &problem = bin_problem_sizes[bin_idx];
            int32_t batch_count = bin_count[bin_idx];
            int32_t bin_start_idx = bin_start[bin_idx];
            int32_t lda = bin_ldm_A[bin_idx];
            int32_t ldb = bin_ldm_B[bin_idx];
            int32_t ldc = bin_ldm_C[bin_idx];

            void const **ptr_A_array = ptr_A_batched.get() + bin_start[bin_idx];
            void const **ptr_B_array = ptr_B_batched.get() + bin_start[bin_idx];
            void **ptr_C_array = ptr_C_batched.get() + bin_start[bin_idx];

            //
            // Initialize the CUTLASS GEMM operator
            //

            // Configure the GEMM arguments
            typename Gemm::EpilogueOutputOp::Params epilogue_op(this->options.alpha, this->options.beta);

            typename Gemm::Arguments arguments{
                cutlass::gemm::GemmUniversalMode::kArray,
                problem,
                batch_count,
                epilogue_op,
                (void const *)ptr_A_array,
                (void const *)ptr_B_array,
                (void const *)ptr_C_array,
                (void *)ptr_C_array,
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
                std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
                return result;
            }

            status = gemm_op();

            if (status != cutlass::Status::kSuccess)
            {
                std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
                return result;
            }
        }

        //
        // Wait for completion
        //

        result.error = cudaDeviceSynchronize();

        if (result.error != cudaSuccess)
        {
            std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
            return result;
        }

        //
        // Construct events
        //

        cudaEvent_t events[2];

        for (auto &event : events)
        {
            result.error = cudaEventCreate(&event);
            if (result.error != cudaSuccess)
            {
                std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
                return -1;
            }
        }

        //
        // Wait for completion
        //

        result.error = cudaDeviceSynchronize();

        if (result.error != cudaSuccess)
        {
            std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
            return result;
        }

        // Record an event at the start of a series of GEMM operations
        result.error = cudaEventRecord(events[0]);
        if (result.error != cudaSuccess)
        {
            std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        //
        // Run profiling loop
        //

        int last_stream_idx = 0;

        for (int iter = 0; iter < this->options.iterations; ++iter)
        {

            for (int bin_idx = 0; bin_idx < int32_t(bin_problem_sizes.size()); ++bin_idx)
            {

                cutlass::gemm::GemmCoord const &problem = bin_problem_sizes[bin_idx];
                int32_t batch_count = bin_count[bin_idx];
                int32_t bin_start_idx = bin_start[bin_idx];
                int32_t lda = bin_ldm_A[bin_idx];
                int32_t ldb = bin_ldm_B[bin_idx];
                int32_t ldc = bin_ldm_C[bin_idx];

                void const **ptr_A_array = ptr_A_batched.get() + bin_start[bin_idx];
                void const **ptr_B_array = ptr_B_batched.get() + bin_start[bin_idx];
                void **ptr_C_array = ptr_C_batched.get() + bin_start[bin_idx];

                last_stream_idx = (bin_idx % effective_streams);

                //
                // Initialize the CUTLASS GEMM operator
                //

                // Configure the GEMM arguments
                typename Gemm::EpilogueOutputOp::Params epilogue_op(this->options.alpha, this->options.beta);

                typename Gemm::Arguments arguments{
                    cutlass::gemm::GemmUniversalMode::kArray,
                    problem,
                    batch_count,
                    epilogue_op,
                    (void const *)ptr_A_array,
                    (void const *)ptr_B_array,
                    (void const *)ptr_C_array,
                    (void *)ptr_C_array,
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
                    std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
                    return result;
                }

                status = gemm_op(cuda_streams[last_stream_idx]);

                if (status != cutlass::Status::kSuccess)
                {
                    std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
                    return result;
                }
            }
        }

        //
        // Stop profiling loop
        //

        // Record an event when the GEMM operations have been launched.
        result.error = cudaEventRecord(events[1]);
        if (result.error != cudaSuccess)
        {
            std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        //
        // Wait for work to be completed
        //

        result.error = cudaDeviceSynchronize();

        if (result.error != cudaSuccess)
        {
            std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
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

        // Compute average runtime and GFLOPs.
        result.runtime_ms = double(runtime_ms) / double(this->options.iterations);
        result.gflops = this->options.gflops(result.runtime_ms / 1000.0);

        //
        // Cleanup
        //

        for (auto event : events)
        {
            (void)cudaEventDestroy(event);
        }

        for (auto stream : cuda_streams)
        {
            if (stream)
            {
                (void)cudaStreamDestroy(stream);
            }
        }

        std::cout << "    " << this->options.problem_bins.size() << " batched GEMMs launched" << std::endl;
        std::cout << std::endl;
        std::cout << "    "
                  << "Batched Runtime: " << result.runtime_ms << " ms" << std::endl;
        std::cout << "    "
                  << "Batched  GFLOPs: " << result.gflops << std::endl;

        std::string provider = "CUTLASS";

        if (this->options.output_file.good())
        {
            this->options.output_file << this->options.output_tag << "," << provider << ",batched,"
                                      << this->options.problem_count << "," << result.runtime_ms << "," << result.gflops << std::endl;
        }

        result.passed = true;
        return result;
    }
};
