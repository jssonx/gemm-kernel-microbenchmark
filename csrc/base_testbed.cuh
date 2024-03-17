#pragma once

#include "helper.cuh"

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