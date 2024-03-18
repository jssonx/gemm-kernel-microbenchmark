#pragma once
#include "helper.cuh"

template <typename Gemm_, cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_>
class TestbedGrouped : BaseTestbed<Gemm_> {
public:
  TestbedGrouped(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
  ): BaseTestbed<Gemm_>(options_, init_A_, init_B_, init_C_, seed_) {}

  // Redefine GEMM with different GroupScheduleMode_
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    typename Gemm_::ElementA,
    typename Gemm_::LayoutA,
    Gemm_::kTransformA,
    Gemm_::kAlignmentA,
    typename Gemm_::ElementB,
    typename Gemm_::LayoutB,
    Gemm_::kTransformB,
    Gemm_::kAlignmentB,
    typename Gemm_::ElementC,
    typename Gemm_::LayoutC,
    typename Gemm_::ElementAccumulator,
    typename Gemm_::OperatorClass,
    typename Gemm_::ArchTag,
    typename Gemm_::ThreadblockShape,
    typename Gemm_::WarpShape,
    typename Gemm_::InstructionShape,
    typename Gemm_::EpilogueOutputOp,
    typename Gemm_::ThreadblockSwizzle,
    Gemm_::kStages,
    GroupScheduleMode_>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  /// Verbose printing of problem sizes
  void print_problem_sizes() {
    std::cout << std::endl;

    // Print groups
    std::cout << this->problem_count() << " groups:\n";

    int32_t idx = 0;
    int64_t total_tiles = 0;

    for (auto const & problem : this->options.problem_sizes) {
      int tiles = Gemm::problem_tile_count(problem);
      total_tiles += tiles;

      std::cout << "  [" << idx << "]: "
        << problem.m() << "-by-" << problem.n() << "-by-" << problem.k()
        << " (" << tiles << " threadblock tiles)" << "\n";

      ++idx;
    }
    std::cout << std::endl;
  }

  /// Sort problems in descending order of problem-K dimension
  void sort_problems() {
    Gemm::sort_problems(this->options.problem_count,
                        this->options.problem_sizes.data(),
                        this->lda_host.data(),
                        this->ldb_host.data(),
                        this->ldc_host.data(),
                        this->ldd_host.data(),
                        this->offset_A.data(),
                        this->offset_B.data(),
                        this->offset_C.data(),
                        this->offset_D.data());
  }

  /// Executes a grouped kernel and measures runtime
  Result profile() {
    std::string sched_mode = this->options.scheduler_mode_to_str.find(GroupScheduleMode_)->second;

    std::cout << std::endl;
    std::cout << "Grouped GEMM (CUTLASS) with mode " << sched_mode << ":\n"
      << "====================================================" << std::endl;

    Result result;

    int threadblock_count = Gemm::sufficient(this->options.problem_sizes.data(), this->options.problem_count);

    // Early exit
    if (!threadblock_count) {
      std::cout << "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel." << std::endl;
      return result;
    }

    result.passed = false;

    // Initialize the problem
    this->allocate();
    if (this->options.sort_problems) {
      sort_problems();
    }
    this->initialize();

    if (this->options.verbose) {
      print_problem_sizes();
    }

    // Configure the GEMM arguments
    typename Gemm::EpilogueOutputOp::Params epilogue_op(this->options.alpha, this->options.beta);

    // Configure GEMM arguments
    typename Gemm::Arguments args(
      this->problem_sizes_device.get(),
      this->problem_count(),
      threadblock_count,
      epilogue_op,
      this->ptr_A.get(),
      this->ptr_B.get(),
      this->ptr_C.get(),
      this->ptr_D.get(),
      this->lda.get(),
      this->ldb.get(),
      this->ldc.get(),
      this->ldd.get(),
      this->options.problem_sizes.data()
    );

    // Initialize the GEMM object
    Gemm gemm;

    size_t workspace_size = gemm.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

    result.status = gemm.initialize(args, workspace.get());

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS Grouped GEMM kernel." << std::endl;
      return result;
    }

    // Run the grouped GEMM object
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
      return result;
    }

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (this->options.reference_check) {
      result.passed = this->verify();
    }

    //
    // Warm-up run of the grouped GEMM object
    //
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
      return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < this->options.iterations; ++iter) {
      gemm();
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(this->options.iterations);
    result.gflops = this->options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    // Optionally profile initialization
    if (this->options.profile_initialization) {
      // Warm up
      gemm.initialize(args, workspace.get());

      auto start_time = std::chrono::high_resolution_clock::now();
      for (int32_t i = 0; i < this->options.iterations; ++i) {
        gemm.initialize(args, workspace.get());
      }
      auto end_time = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> duration = end_time - start_time;
      duration /= double(this->options.iterations);
      result.initialization_time_ms = duration.count();
    }

    int64_t total_tiles = Gemm::group_tile_count(args);
    std::cout << "    " << total_tiles << " total threadblock tiles." << std::endl;

    std::cout << std::endl;
    std::cout << "    " << "Grouped Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "Grouped  GFLOPs: " << result.gflops << std::endl;
    if (this->options.profile_initialization) {
      std::cout << "    " << "Init    Runtime: " << result.initialization_time_ms << " ms" << std::endl;
    }

    if (this->options.output_file.good()) {
      this->options.output_file << this->options.output_tag << ",CUTLASS,grouped-" << sched_mode << ","
        << this->options.problem_count << "," << result.runtime_ms << "," << result.gflops << std::endl;
    }

    std::cout << "\nPassed\n";

    return result;
  }
};
