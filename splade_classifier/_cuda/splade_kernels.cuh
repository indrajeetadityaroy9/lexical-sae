// Requires CUDA CC >= 7.0. FLOPS reg: sum_v (mean_b |w[b,v]|)^2
#ifndef SPLADE_KERNELS_CUH
#define SPLADE_KERNELS_CUH

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE_AGGREGATE = 256;
constexpr int BLOCK_SIZE_FLOPS = 256;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float fast_log1p(float x) { return __logf(1.0f + x); }

__device__ __forceinline__ float splade_activation(float x) {
    return fast_log1p(fmaxf(x, 0.0f));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE, warp_id = threadIdx.x / WARP_SIZE;
    val = warp_reduce_sum(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared_mem[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

// out[b,v] = max_s(log1p(relu(logits[b,s,v])) * mask[b,s])
__global__ void splade_aggregate_kernel(
    const float* __restrict__ logits, const float* __restrict__ attention_mask,
    float* __restrict__ output, int batch_size, int seq_len, int vocab_size);

__global__ void splade_aggregate_kernel_vectorized(
    const float4* __restrict__ logits, const float* __restrict__ attention_mask,
    float4* __restrict__ output, int batch_size, int seq_len, int vocab_size_div4);

__global__ void flops_reg_column_sum_kernel(
    const float* __restrict__ activations, float* __restrict__ col_sums,
    int batch_size, int vocab_size);

__global__ void flops_reg_final_kernel(
    const float* __restrict__ col_sums, float* __restrict__ loss,
    int vocab_size, float batch_size_inv);

__global__ void flops_reg_fused_kernel(
    const float* __restrict__ activations, float* __restrict__ loss,
    int batch_size, int vocab_size);

// DF-weighted FLOPS: sum_v (df_weight[v] * mean_b |w[b,v]|)^2
__global__ void df_flops_reg_column_sum_kernel(
    const float* __restrict__ activations, const float* __restrict__ df_weights,
    float* __restrict__ col_sums, int batch_size, int vocab_size);

__global__ void df_flops_reg_fused_kernel(
    const float* __restrict__ activations, const float* __restrict__ df_weights,
    float* __restrict__ loss, int batch_size, int vocab_size);

#ifdef __cplusplus
extern "C" {
#endif

void launch_splade_aggregate(const float* logits, const float* attention_mask,
    float* output, int batch_size, int seq_len, int vocab_size, cudaStream_t stream = 0);

void launch_flops_reg(const float* activations, float* loss, float* workspace,
    int batch_size, int vocab_size, cudaStream_t stream = 0);

void launch_df_flops_reg(const float* activations, const float* df_weights,
    float* loss, float* workspace, int batch_size, int vocab_size, cudaStream_t stream = 0);

size_t get_flops_reg_workspace_size(int vocab_size);

#ifdef __cplusplus
}
#endif

#endif // SPLADE_KERNELS_CUH
