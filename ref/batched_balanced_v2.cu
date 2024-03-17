#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"

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

  int const count_A = batch_count * lda * k;
  int const count_B = ldb * n;
  int const count_C = batch_count * ldc * n;

  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);

  float alpha = 1.0f;
  float beta = 2.0f;

  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);

  initialize_matrix(host_A, m * batch_count, k, 0);
  initialize_matrix(host_B, k * batch_count, n, 1);
  initialize_matrix(host_C, m * batch_count, n, 2);

  result = cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
  result = cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
  result = cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);

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

  result = cudaMalloc(&ptr_A, batch_count * sizeof(float *));
  if (result != cudaSuccess)
  {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&ptr_B, batch_count * sizeof(float *));
  if (result != cudaSuccess)
  {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }
  result = cudaMalloc(&ptr_C, batch_count * sizeof(float *));
  if (result != cudaSuccess)
  {
    std::cerr << "cudaMalloc result = " << result << std::endl;
    return result;
  }

  result = cudaMemcpy(ptr_A, host_ptr_A.data(), batch_count * sizeof(float *), cudaMemcpyHostToDevice);
  if (result != cudaSuccess)
  {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(ptr_B, host_ptr_B.data(), batch_count * sizeof(float *), cudaMemcpyHostToDevice);
  if (result != cudaSuccess)
  {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }
  result = cudaMemcpy(ptr_C, host_ptr_C.data(), batch_count * sizeof(float *), cudaMemcpyHostToDevice);
  if (result != cudaSuccess)
  {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  result = cutlass_array_sgemm(m, n, k, alpha, ptr_A, lda, ptr_B, ldb, ptr_C, ldc, beta, batch_count);
  if (result != cudaSuccess)
  {
    std::cerr << "cutlass_array_sgemm failed" << std::endl;
    return result;
  }

  std::vector<float> result_C(count_C);
  result = cudaMemcpy(result_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost);
  if (result != cudaSuccess)
  {
    std::cerr << "cudaMemcpy result = " << result << std::endl;
    return result;
  }

  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);

  result = strided_batched_gemm_nn_reference(m, n, k, alpha, ref_A, lda, batch_stride_A, ref_B, ldb, batch_stride_B, ref_C, ldc, batch_stride_C, beta, batch_count);

  if (ref_C != result_C)
  {
    std::cout << "CUTLASS batched GEMM does not run correctly" << std::endl;
    return cudaErrorUnknown;
  }

  // free memory
  result = cudaFree(ptr_A);
  if (result != cudaSuccess)
  {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(ptr_B);
  if (result != cudaSuccess)
  {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }
  result = cudaFree(ptr_C);
  if (result != cudaSuccess)
  {
    std::cerr << "cudaFree result = " << result << std::endl;
    return result;
  }

  return result;
}

int main()
{
  cudaError_t result = cudaSuccess;

  int const m = 12;
  int const n = 768 * 4;
  int const k = 768;
  int const batch_count = 8;

  int const count_A = batch_count * m * k;
  int const count_B = batch_count * k * n;
  int const count_C = batch_count * m * n;

  float *A;
  float *B;
  float *C;

  result = cudaMalloc(&A, count_A * sizeof(float));
  result = cudaMalloc(&B, count_B * sizeof(float));
  result = cudaMalloc(&C, count_C * sizeof(float));

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

  float *B2;
  float *C2;

  result = cudaMalloc(&B2, count_B2 * sizeof(float));
  result = cudaMalloc(&C2, count_C2 * sizeof(float));

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

  result = cudaFree(A);
  result = cudaFree(B);
  result = cudaFree(C);
  result = cudaFree(B2);
  result = cudaFree(C2);

  return result == cudaSuccess ? 0 : -1;
}