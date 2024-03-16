#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

#include "../cuda_check.cuh"

#pragma warning(disable : 4503)

cudaError_t cutlass_array_sgemm(
    int m,
    int n,
    int k,
    float alpha,
    float const *const *A,
    int lda,
    float const *const *B,
    int ldb,
    float *const *C,
    int ldc,
    float beta,
    int batch_count)
{

  using Gemm = cutlass::gemm::device::GemmArray<
      float, cutlass::layout::ColumnMajor,
      float, cutlass::layout::ColumnMajor,
      float, cutlass::layout::ColumnMajor>;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({{m, n, k},
                                    A,
                                    lda,
                                    B,
                                    ldb,
                                    C,
                                    ldc,
                                    C,
                                    ldc,
                                    {alpha, beta},
                                    batch_count});

  if (status != cutlass::Status::kSuccess)
  {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

template <typename T>
cudaError_t strided_batched_gemm_nn_reference(
    int m,
    int n,
    int k,
    T alpha,
    std::vector<T> const &A,
    int lda,
    long long int batch_stride_A,
    std::vector<T> const &B,
    int ldb,
    long long int batch_stride_B,
    std::vector<T> &C,
    int ldc,
    long long int batch_stride_C,
    T beta,
    int batch_count)
{
  /*
  strided batched gemm NN
  */

  cudaError_t result = cudaSuccess;

  if (A.size() < lda * k * batch_count)
  {
    std::cout << "the size of A is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  if (B.size() < ldb * n)
  {
    std::cout << "the size of B is too small" << std::endl;
    return cudaErrorInvalidValue;
  }
  if (C.size() < ldc * n * batch_count)
  {
    std::cout << "the size of C is too small" << std::endl;
    return cudaErrorInvalidValue;
  }

  for (int batch_idx = 0; batch_idx < batch_count; batch_idx++)
  {
    for (int n_idx = 0; n_idx < n; n_idx++)
    {
      for (int m_idx = 0; m_idx < m; m_idx++)
      {
        T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
          accum += alpha * A[batch_idx * batch_stride_A + k_idx * lda + m_idx] * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
        }
        C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
      }
    }
  }

  return result;
}

template <typename T>
void initialize_matrix(std::vector<T> &matrix, int num_rows, int num_cols, int seed = 0)
{
  int const kRange = 8;
  srand(seed);

  for (int col_idx = 0; col_idx < num_cols; col_idx++)
  {
    for (int row_idx = 0; row_idx < num_rows; row_idx++)
    {
      matrix[row_idx + col_idx * num_rows] = static_cast<T>((rand() % kRange) - (kRange / 2));
    }
  }
}

cudaError_t run_batched_gemm(
    float *A,
    float *B,
    float *C,
    int m,
    int n,
    int k,
    int batch_count)
{
  cudaError_t result = cudaSuccess;

  int const lda = m;
  int const ldb = k * batch_count;
  int const ldc = m;

  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);

  float alpha = 1.0f;
  float beta = 2.0f;

  std::vector<float *> host_ptr_A(batch_count);
  std::vector<float *> host_ptr_B(batch_count);
  std::vector<float *> host_ptr_C(batch_count);

  for (size_t b_idx = 0; b_idx < batch_count; b_idx++)
  {
    host_ptr_A[b_idx] = A + b_idx * batch_stride_A;
    host_ptr_B[b_idx] = B + b_idx * batch_stride_B;
    host_ptr_C[b_idx] = C + b_idx * batch_stride_C;
  }

  float const **ptr_A;
  float const **ptr_B;
  float **ptr_C;

  CHECK(cudaMalloc(&ptr_A, batch_count * sizeof(float *)));
  CHECK(cudaMalloc(&ptr_B, batch_count * sizeof(float *)));
  CHECK(cudaMalloc(&ptr_C, batch_count * sizeof(float *)));

  CHECK(cudaMemcpy(ptr_A, host_ptr_A.data(), batch_count * sizeof(float *), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(ptr_B, host_ptr_B.data(), batch_count * sizeof(float *), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(ptr_C, host_ptr_C.data(), batch_count * sizeof(float *), cudaMemcpyHostToDevice));

  CHECK(cutlass_array_sgemm(m, n, k, alpha, ptr_A, lda, ptr_B, ldb, ptr_C, ldc, beta, batch_count));

  // free memory
  CHECK(cudaFree(ptr_A));
  CHECK(cudaFree(ptr_B));
  CHECK(cudaFree(ptr_C));

  return result;
}

int main()
{
  cudaError_t result = cudaSuccess;

  int const m = 12;
  int const n = 768 * 4;
  int const k = 768;
  int const batch_count = 8; // The 'batch_count' represents the number of experts in SparseMLP

  int const count_A = batch_count * m * k;
  int const count_B = batch_count * k * n;
  int const count_C = batch_count * m * n;

  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);

  initialize_matrix(host_A, m * batch_count, k, 0);
  initialize_matrix(host_B, k * batch_count, n, 1);
  initialize_matrix(host_C, m * batch_count, n, 2);

  float *A;
  float *B;
  float *C;

  CHECK(cudaMalloc(&A, count_A * sizeof(float)));
  CHECK(cudaMalloc(&B, count_B * sizeof(float)));
  CHECK(cudaMalloc(&C, count_C * sizeof(float)));

  CHECK(cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice));

  result = run_batched_gemm(A, B, C, m, n, k, batch_count);
  if (result == cudaSuccess)
  {
    std::cout << "First batched GEMM passed." << std::endl;
  }
  else
  {
    std::cerr << "First batched GEMM failed." << std::endl;
    return result;
  }

  int const m2 = m;
  int const n2 = k;
  int const k2 = n;

  int const count_B2 = batch_count * k2 * n2;
  int const count_C2 = batch_count * m2 * n2;

  std::vector<float> host_B2(count_B2);
  std::vector<float> host_C2(count_C2);

  initialize_matrix(host_B2, k2 * batch_count, n2, 3);
  initialize_matrix(host_C2, m2 * batch_count, n2, 4);

  float *B2;
  float *C2;

  CHECK(cudaMalloc(&B2, count_B2 * sizeof(float)));
  CHECK(cudaMalloc(&C2, count_C2 * sizeof(float)));

  CHECK(cudaMemcpy(B2, host_B2.data(), count_B2 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(C2, host_C2.data(), count_C2 * sizeof(float), cudaMemcpyHostToDevice));

  result = run_batched_gemm(C, B2, C2, m2, n2, k2, batch_count);
  if (result == cudaSuccess)
  {
    std::cout << "Second batched GEMM passed." << std::endl;
  }
  else
  {
    std::cerr << "Second batched GEMM failed." << std::endl;
    return result;
  }

  std::vector<float> result_C(count_C);
  std::vector<float> result_C2(count_C2);
  CHECK(cudaMemcpy(result_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(result_C2.data(), C2, count_C2 * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);
  std::vector<float> ref_B2(host_B2);
  std::vector<float> ref_C2(host_C2);

  CHECK(strided_batched_gemm_nn_reference(m, n, k, 1.0f, ref_A, m, m * k, ref_B, k * batch_count, k, ref_C, m, m * n, 2.0f, batch_count));
  if (ref_C != result_C)
  {
    std::cout << "CUTLASS batched GEMM does not run correctly" << std::endl;
    return cudaErrorUnknown;
  }

  CHECK(strided_batched_gemm_nn_reference(m2, n2, k2, 1.0f, result_C, m2, m2 * k2, ref_B2, k2 * batch_count, k2, ref_C2, m2, m2 * n2, 2.0f, batch_count));
  if (ref_C2 != result_C2)
  {
    std::cout << "CUTLASS batched GEMM does not run correctly" << std::endl;
    return cudaErrorUnknown;
  }

  CHECK(cudaFree(A));
  CHECK(cudaFree(B));
  CHECK(cudaFree(C));
  CHECK(cudaFree(B2));
  CHECK(cudaFree(C2));

  return result == cudaSuccess ? 0 : -1;
}