// This file was automatically generated by the CUTLASS 3.4.1 Python interface (https://github.com/nvidia/cutlass/python)

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>

// CUDA forward declarations
std::vector<at::Tensor> grouped_gemm_kernel(const std::vector<at::Tensor>& A, const std::vector<at::Tensor>& B, at::optional<const std::vector<at::Tensor>> C=at::nullopt, float alpha=1.f, float beta=0.f);

// C++ interface
std::vector<at::Tensor> grouped_gemm(const std::vector<at::Tensor>& A, const std::vector<at::Tensor>& B, at::optional<const std::vector<at::Tensor>> C=at::nullopt, float alpha=1.f, float beta=0.f) {
  return grouped_gemm_kernel(A, B, C, alpha, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", py::overload_cast<const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, at::optional<const std::vector<at::Tensor>>, float, float>(&grouped_gemm),
        py::arg("A"), py::arg("B"), py::arg("C") = nullptr, py::arg("alpha") = 1.f, py::arg("beta") = 0.f);
}
