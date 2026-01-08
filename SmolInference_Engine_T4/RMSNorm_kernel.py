import torch
import time
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float warpReduceSum(float val){
    for(int offset=16;offset>=1;offset/=2){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

__global__ void rmsnorm_smollm_kernel(const float* __restrict__ input,const float* __restrict__ weights,int hidden_dim,float epsilon,float* __restrict__ output){
    int tid=threadIdx.x;
    int rowidx=blockIdx.x;
    
    const float* row_input=input+(rowidx*hidden_dim);
    float* row_output=output+(rowidx*hidden_dim);

    float sum_sq=0.0f;
    for(int i=tid;i<hidden_dim;i+=blockDim.x){
        float val=row_input[i];
        sum_sq+=val*val;
    }

    sum_sq=warpReduceSum(sum_sq);

    __shared__ float shared_warp_sums[32];
    int lane=tid%32;
    int warp_id=tid/32;

    if(lane==0){
        shared_warp_sums[warp_id]=sum_sq;
    }
    __syncthreads();

    float final_sum_sq=0.0f;
    if(warp_id==0){
        if(tid<32){
             final_sum_sq=shared_warp_sums[lane];
        }
        final_sum_sq=warpReduceSum(final_sum_sq);
    }
    
    if(tid==0){
        float mean=final_sum_sq/hidden_dim;
        shared_warp_sums[0]=rsqrtf(mean+epsilon);
    }
    __syncthreads();
    
    float inv_rms=shared_warp_sums[0];
    
    for(int i=tid;i<hidden_dim;i+=blockDim.x){
        row_output[i]=row_input[i]*inv_rms*weights[i];
   }
}

torch::Tensor run_rmsnorm(torch::Tensor input,torch::Tensor weights,float epsilon){
    auto output=torch::empty_like(input);
    int batch_size=input.size(0);
    int hidden_dim=input.size(1);
    
    int threads=1024;
    int blocks=batch_size;
    
    rmsnorm_smollm_kernel<<<blocks,threads>>>(input.data_ptr<float>(),weights.data_ptr<float>(),hidden_dim,epsilon,output.data_ptr<float>());
    
    return output;
}
"""

cpp_source="torch::Tensor run_rmsnorm(torch::Tensor input,torch::Tensor weights,float epsilon);"

rmsnorm_module=load_inline(
    name='smollm_rmsnorm_compact',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['run_rmsnorm'],
    with_cuda=True,
    extra_cuda_cflags=["-O3"]
)

# --- BENCHMARK ---
def benchmark_smollm_rmsnorm():
    HIDDEN_DIM=2048
    BATCH_SIZE=4096
    
    print(f"🚀 Benchmarking (Dim={HIDDEN_DIM})...")
    
    x=torch.randn(BATCH_SIZE,HIDDEN_DIM,device='cuda',dtype=torch.float32)
    w=torch.ones(HIDDEN_DIM,device='cuda',dtype=torch.float32)
    
    # Warmup
    rmsnorm_module.run_rmsnorm(x,w,1e-6)
        
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(1000):
        var=x.pow(2).mean(-1,keepdim=True)
        y_py=x*torch.rsqrt(var+1e-6)*w
    torch.cuda.synchronize()
    py_time=time.time()-t0
    
    torch.cuda.synchronize()
    t1=time.time()
    for _ in range(1000):
        y_custom=rmsnorm_module.run_rmsnorm(x,w,1e-6)
    torch.cuda.synchronize()
    custom_time=time.time()-t1
    
    print(f"\n🔴 PyTorch Time: {py_time:.4f}s")
    print(f"🟢 Custom Kernel: {custom_time:.4f}s")
    print(f"⚡ Speedup: {py_time/custom_time:.2f}x")

benchmark_smollm_rmsnorm()