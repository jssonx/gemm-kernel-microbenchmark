#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

/// Result structure
struct Result
{

  double runtime_ms;
  double initialization_time_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
      double runtime_ms = 0,
      double initialization_time_ms = 0,
      double gflops = 0,
      cutlass::Status status = cutlass::Status::kSuccess,
      cudaError_t error = cudaSuccess) : runtime_ms(runtime_ms), initialization_time_ms(initialization_time_ms), gflops(gflops),
                                         status(status), error(error), passed(true) {}
};

/// Hash function for cutlass::gemm::GemmCoord
struct HashGemmCoord
{
  size_t operator()(cutlass::gemm::GemmCoord const &problem) const
  {
    std::hash<int> hasher;
    return (hasher(problem.m() * 3)) ^ (hasher(1 + problem.n() * 5)) ^ (hasher(2 + problem.k() * 7));
  }
};

// Command line options parsing
struct Options
{

  bool error;
  bool reference_check;
  bool profile_initialization;
  bool sort_problems;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes;

  // problem size bins
  std::unordered_map<
      cutlass::gemm::GemmCoord,
      std::vector<int32_t>,
      HashGemmCoord>
      problem_bins;

  int alignment;
  int problem_count;
  int iterations;
  int cuda_streams;
  float alpha;
  float beta;
  std::string benchmark_path;

  std::string output_tag;
  std::ofstream output_file;

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;
  std::vector<GroupScheduleMode> scheduler_modes;

  std::unordered_map<std::string, GroupScheduleMode>
      str_to_scheduler_mode = {
          {"kDeviceOnly", GroupScheduleMode::kDeviceOnly},
          {"kHostPrecompute", GroupScheduleMode::kHostPrecompute}};

  struct GroupScheduleModeHash
  {
    size_t operator()(GroupScheduleMode m) const
    {
      return static_cast<size_t>(m);
    }
  };

  std::unordered_map<GroupScheduleMode, std::string, GroupScheduleModeHash>
      scheduler_mode_to_str = {
          {GroupScheduleMode::kDeviceOnly, "kDeviceOnly"},
          {GroupScheduleMode::kHostPrecompute, "kHostPrecompute"}};

  std::vector<GroupScheduleMode> all_scheduler_modes = {GroupScheduleMode::kDeviceOnly, GroupScheduleMode::kHostPrecompute};

  //
  // Methods
  //

  Options() : error(false),
              alignment(8),
              reference_check(true),
              profile_initialization(false),
              sort_problems(false),
              problem_count(15),
              iterations(20),
              cuda_streams(0),
              alpha(1),
              beta(0),
              scheduler_modes({GroupScheduleMode::kDeviceOnly})
  {
  }

  // Parses the command line
  void parse()
  {
    // Initialize the problems
    randomize_problems();

    // Post-process the problem sizes
    bin_problems();
  }

  void randomize_problems()
  {
    // Default problem size
    int default_m = 256;
    int default_n = 256;
    int default_k = 256;

    problem_sizes.reserve(problem_count);

    for (int i = 0; i < problem_count; ++i)
    {
      int m = default_m;
      int n = default_n;
      int k = default_k;

      // Generate random problem sizes
      m = alignment * ((rand() % 256) + 1);
      n = alignment * ((rand() % 256) + 1);
      k = alignment * ((rand() % 256) + 1);

      cutlass::gemm::GemmCoord problem(m, n, k);

      problem_sizes.push_back(problem);
    }
  }

  /// Post processes the problems
  void bin_problems()
  {

    problem_bins.clear();

    problem_count = int(problem_sizes.size());

    //
    // Insert the problem sizes into a sorted container class. This is *NOT* necessary
    // to run the CUTLASS kernel, but it enables the execution of cublas's batched GEMM.
    //
    for (int i = 0; i < int(problem_sizes.size()); ++i)
    {
      auto it = problem_bins.find(problem_sizes.at(i));
      if (it == problem_bins.end())
      {
        problem_bins.insert({problem_sizes.at(i), std::vector<int32_t>({i})});
      }
      else
      {
        it->second.push_back(i);
      }
    }
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {

    // Number of real-valued multiply-adds
    int64_t fmas = int64_t();

    for (auto const &problem : problem_sizes)
    {
      fmas += problem.product();
    }

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};