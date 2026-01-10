#pragma once
#include <torch/extension.h>

// Computes RMSNorm: output = input * rsqrt(mean(input^2) + eps) * weight
torch::Tensor run_rmsnorm(torch::Tensor input, torch::Tensor weight, float epsilon);

// Computes SwiGLU: output = (gate * sigmoid(gate)) * up
void apply_swiglu(torch::Tensor gate, torch::Tensor up, torch::Tensor output);

// Computes RoPE: Rotates query/key vectors based on position
void apply_rope(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);