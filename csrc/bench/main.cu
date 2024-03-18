#include "helper.cuh"
#include "grouped.cuh"
#include "batched_bin.cuh"
#include "batched_nopadding.cuh"
#include "batched_padding.cuh"

int main(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {

    //
    // This example requires an NVIDIA Ampere-architecture GPU.
    //

    std::cout
      << "CUTLASS's Grouped GEMM example requires a GPU of NVIDIA's Ampere Architecture or "
      << "later (compute capability 80 or greater).\n";

    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Define the Grouped and Batched GEMM types
  //

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_align8
  using GemmBatched = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementOutput,   LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4
  >;

  // Define a grouped GEMM kernel with all template parameters set except
  // for scheduling mode. This will be used as the template for all scheduling
  // modes executed.
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    ElementB,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
    // This parameter is passed in at present to match the APIs of other kernels. The parameter
    // is unused within the kernel.
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Profile it
  //

  TestbedBatched<GemmBatched> testbed_batched(options);
  Result result = testbed_batched.profile();
  if (result.error) {
    return 1;
  }

  TestbedBatched_NP<GemmBatched> testbed_batched_np(options);
  result = testbed_batched_np.profile();
  if (result.error) {
    return 1;
  }

  TestbedBatched_P<GemmBatched> testbed_batched_p(options);
  result = testbed_batched_p.profile();
  if (result.error) {
    return 1;
  }

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;
  for (GroupScheduleMode mode : options.scheduler_modes) {
    Result result;
    switch (mode) {
      case GroupScheduleMode::kDeviceOnly:
        {
          TestbedGrouped<GemmGrouped, GroupScheduleMode::kDeviceOnly> runner(options);
          result = runner.profile();
          break;
        }
      case GroupScheduleMode::kHostPrecompute:
        {
          TestbedGrouped<GemmGrouped, GroupScheduleMode::kHostPrecompute> runner(options);
          result = runner.profile();
          break;
        }
    }

    if (result.error != cudaSuccess) {
      return 1;
    }

    // Override verbose flag to avoid printing duplicate information for each scheduling mode
    options.verbose = false;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////