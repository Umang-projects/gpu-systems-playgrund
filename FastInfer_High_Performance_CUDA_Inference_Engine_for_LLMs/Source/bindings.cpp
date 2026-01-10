#include <torch/extension.h>
#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &run_rmsnorm, "RMSNorm FP16");
    m.def("swiglu", &apply_swiglu, "SwiGLU FP16");
    m.def("rope", &apply_rope, "RoPE FP16");
    m.def("gqa_scores", &gqa_dot_product, "GQA Dot Product");
}