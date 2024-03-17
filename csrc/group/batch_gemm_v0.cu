#include "../helper.cuh"

template <typename Gemm>
class BaseTestbed
{
public:
  //
  // Type definitions
  //

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  using MatrixCoord = typename LayoutC::TensorCoord;

  //
  // Data members
  //

  Options &options;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint32_t seed;

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;

  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<int64_t> lda_host;
  std::vector<int64_t> ldb_host;
  std::vector<int64_t> ldc_host;
  std::vector<int64_t> ldd_host;

  cutlass::DeviceAllocation<int64_t> lda;
  cutlass::DeviceAllocation<int64_t> ldb;
  cutlass::DeviceAllocation<int64_t> ldc;
  cutlass::DeviceAllocation<int64_t> ldd;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D;

  cutlass::DeviceAllocation<ElementA *> ptr_A;
  cutlass::DeviceAllocation<ElementB *> ptr_B;
  cutlass::DeviceAllocation<ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementC *> ptr_D;

  BaseTestbed(
      Options &options_,
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      uint32_t seed_ = 3080) : options(options_), init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) {}

  int problem_count() const
  {
    return options.problem_count;
  }

  /// Helper to initialize a tensor view
  template <typename Element>
  void initialize_tensor(
      Element *ptr,
      size_t capacity,
      cutlass::Distribution::Kind dist_kind,
      uint32_t seed);

  /// Allocates device-side data
  void allocate();

  /// Initializes device-side data
  void initialize();

  /// Verifies the result is a GEMM
  bool verify();
};

template <typename Gemm>
void BaseTestbed<Gemm>::allocate()
{
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;

  lda_host.resize(problem_count());
  ldb_host.resize(problem_count());
  ldc_host.resize(problem_count());
  ldd_host.resize(problem_count());

  for (int32_t i = 0; i < problem_count(); ++i)
  {

    auto problem = options.problem_sizes.at(i);

    lda_host.at(i) = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    ldb_host.at(i) = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    ldc_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    ldd_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);

    int64_t elements_A = problem.m() * problem.k();
    int64_t elements_B = problem.k() * problem.n();
    int64_t elements_C = problem.m() * problem.n();
    int64_t elements_D = problem.m() * problem.n();

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
  }

  lda.reset(problem_count());
  ldb.reset(problem_count());
  ldc.reset(problem_count());
  ldd.reset(problem_count());

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
}

template <typename Gemm>
void BaseTestbed<Gemm>::initialize()
{
  problem_sizes_device.reset(problem_count());
  problem_sizes_device.copy_from_host(options.problem_sizes.data());

  lda.copy_from_host(lda_host.data());
  ldb.copy_from_host(ldb_host.data());
  ldc.copy_from_host(ldc_host.data());
  ldd.copy_from_host(ldd_host.data());

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(problem_count());
  std::vector<ElementB *> ptr_B_host(problem_count());
  std::vector<ElementC *> ptr_C_host(problem_count());
  std::vector<ElementC *> ptr_D_host(problem_count());

  for (int32_t i = 0; i < problem_count(); ++i)
  {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
  }

  ptr_A.reset(problem_count());
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(problem_count());
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(problem_count());
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(problem_count());
  ptr_D.copy_from_host(ptr_D_host.data());

  //
  // Initialize the problems of the workspace
  //

  initialize_tensor(block_A.get(), block_A.size(), init_A, seed * 2021);
  initialize_tensor(block_B.get(), block_B.size(), init_B, seed * 2022);
  initialize_tensor(block_C.get(), block_C.size(), init_C, seed * 2023);

  cutlass::reference::device::BlockFillSequential(
      block_D.get(), block_D.size(), ElementC(), ElementC());
}

template <typename Gemm>
template <typename Element>
void BaseTestbed<Gemm>::initialize_tensor(
    Element *ptr,
    size_t capacity,
    cutlass::Distribution::Kind dist_kind,
    uint32_t seed)
{

  if (dist_kind == cutlass::Distribution::Uniform)
  {

    Element scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;
    int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

    if (bits_input == 1)
    {
      scope_max = 2;
      scope_min = 0;
    }
    else if (bits_input <= 8)
    {
      scope_max = 2;
      scope_min = -2;
    }
    else if (bits_output == 16)
    {
      if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16)
      {
        scope_max = 5;
        scope_min = -5;
      }
      else
      {
        scope_max = 8;
        scope_min = -8;
      }
    }
    else
    {
      scope_max = 8;
      scope_min = -8;
    }

    cutlass::reference::device::BlockFillRandomUniform(
        ptr, capacity, seed, scope_max, scope_min, 0);
  }
  else if (dist_kind == cutlass::Distribution::Gaussian)
  {

    cutlass::reference::device::BlockFillRandomGaussian(
        ptr, capacity, seed, Element(), Element(0.5f));
  }
  else if (dist_kind == cutlass::Distribution::Sequential)
  {

    // Fill with increasing elements
    cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(1), Element());
  }
  else
  {

    // Fill with all 1s
    cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(), Element(1));
  }
}

template <typename Gemm>
bool BaseTestbed<Gemm>::verify()
{

  bool passed = true;

  for (int32_t i = 0; i < problem_count(); ++i)
  {
    cutlass::gemm::GemmCoord problem = options.problem_sizes.at(i);

    LayoutA layout_A(lda_host.at(i));
    LayoutB layout_B(ldb_host.at(i));
    LayoutC layout_C(ldc_host.at(i));
    LayoutC layout_D(ldd_host.at(i));

    MatrixCoord extent_A{problem.m(), problem.k()};
    MatrixCoord extent_B{problem.k(), problem.n()};
    MatrixCoord extent_C{problem.m(), problem.n()};

    cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get() + offset_A.at(i), layout_A, extent_A);
    cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get() + offset_B.at(i), layout_B, extent_B);
    cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get() + offset_C.at(i), layout_C, extent_C);

    cutlass::DeviceAllocation<ElementC> block_Ref(layout_D.capacity(extent_C));
    cutlass::TensorView<ElementC, LayoutC> view_Ref_device(block_Ref.get(), layout_D, extent_C);

    // Reference GEMM
    cutlass::reference::device::GemmComplex<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementCompute, ElementAccumulator>(
        problem,
        options.alpha,
        view_A,
        Gemm::kTransformA,
        view_B,
        Gemm::kTransformB,
        options.beta,
        view_C,
        view_Ref_device,
        ElementAccumulator(0));

    // Copy to host memory
    std::vector<ElementC> matrix_D(layout_D.capacity(extent_C));
    std::vector<ElementC> matrix_Ref(layout_D.capacity(extent_C));

    cutlass::device_memory::copy_to_host(matrix_D.data(), block_D.get() + offset_D.at(i), matrix_D.size());
    cutlass::device_memory::copy_to_host(matrix_Ref.data(), block_Ref.get(), matrix_D.size());

    cutlass::TensorView<ElementC, LayoutC> view_D(matrix_D.data(), layout_D, extent_C);
    cutlass::TensorView<ElementC, LayoutC> view_Ref(matrix_Ref.data(), layout_D, extent_C);

    // Reference check
    passed = cutlass::reference::host::TensorEquals(view_D, view_Ref);

    if (!passed)
    {
      std::cerr << "\n***\nError - problem " << i << " failed the QA check\n***\n"
                << std::endl;
      return passed;
    }
  }

  return passed;
}

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

  void print_problem_sizes();

  /// Executes a batched kernel and measures runtime
  Result profile();
};

template <typename Gemm>
void TestbedBatched<Gemm>::print_problem_sizes()
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

template <typename Gemm>
Result TestbedBatched<Gemm>::profile()
{
  std::cout << "Batched GEMM:\n"
            << "====================================================" << std::endl;

  Result result;
  result.passed = false;

  // Initialize the problem
  this->allocate();
  this->initialize();

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

  result.passed = true;
  return result;
}

int main(int argc, char const **args)
{
  // Parse options
  Options options;
  options.parse();

  // Define the types
  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_align8
  typedef cutlass::gemm::device::GemmUniversal<
      cutlass::half_t, cutlass::layout::ColumnMajor,
      cutlass::half_t, cutlass::layout::ColumnMajor,
      cutlass::half_t, cutlass::layout::ColumnMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          cutlass::half_t,
          128 / cutlass::sizeof_bits<cutlass::half_t>::value,
          float,
          float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      4>
      GemmBatched;

  // Profile it
  TestbedBatched<GemmBatched> testbed_batched(options);
  Result result = testbed_batched.profile();
  if (result.error)
  {
    return 1;
  }

  return 0;
}