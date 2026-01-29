/* FLOPS reg uses atomicAdd which is non-deterministic (<1e-6 variance, no convergence impact).
 * For deterministic training: CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms(True) */

#include "splade_kernels.cuh"
#include <cfloat>

__global__ void splade_aggregate_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ attention_mask,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int vocab_size
) {
    int batch_idx = blockIdx.x;
    int vocab_start = blockIdx.y * blockDim.x;
    int vocab_idx = vocab_start + threadIdx.x;

    if (batch_idx >= batch_size || vocab_idx >= vocab_size) return;

    const float* batch_logits = logits + batch_idx * seq_len * vocab_size;
    const float* batch_mask = attention_mask + batch_idx * seq_len;
    float max_val = -FLT_MAX;

    #pragma unroll 4
    for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        float mask_val = batch_mask[seq_idx];
        if (mask_val > 0.0f) {
            float logit = batch_logits[seq_idx * vocab_size + vocab_idx];
            float activated = splade_activation(logit);
            max_val = fmaxf(max_val, activated);
        }
    }

    if (max_val == -FLT_MAX) max_val = 0.0f;
    output[batch_idx * vocab_size + vocab_idx] = max_val;
}

// Requires vocab_size % 4 == 0
__global__ void splade_aggregate_kernel_vectorized(
    const float4* __restrict__ logits,
    const float* __restrict__ attention_mask,
    float4* __restrict__ output,
    int batch_size,
    int seq_len,
    int vocab_size_div4
) {
    int batch_idx = blockIdx.x;
    int vocab_start = blockIdx.y * blockDim.x;
    int vocab_idx = vocab_start + threadIdx.x;

    if (batch_idx >= batch_size || vocab_idx >= vocab_size_div4) return;

    const float4* batch_logits = logits + batch_idx * seq_len * vocab_size_div4;
    const float* batch_mask = attention_mask + batch_idx * seq_len;
    float4 max_val = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

    #pragma unroll 4
    for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        float mask_val = batch_mask[seq_idx];
        if (mask_val > 0.0f) {
            float4 logit = batch_logits[seq_idx * vocab_size_div4 + vocab_idx];
            float4 activated;
            activated.x = splade_activation(logit.x);
            activated.y = splade_activation(logit.y);
            activated.z = splade_activation(logit.z);
            activated.w = splade_activation(logit.w);
            max_val.x = fmaxf(max_val.x, activated.x);
            max_val.y = fmaxf(max_val.y, activated.y);
            max_val.z = fmaxf(max_val.z, activated.z);
            max_val.w = fmaxf(max_val.w, activated.w);
        }
    }

    if (max_val.x == -FLT_MAX) max_val.x = 0.0f;
    if (max_val.y == -FLT_MAX) max_val.y = 0.0f;
    if (max_val.z == -FLT_MAX) max_val.z = 0.0f;
    if (max_val.w == -FLT_MAX) max_val.w = 0.0f;
    output[batch_idx * vocab_size_div4 + vocab_idx] = max_val;
}

// col_sums[v] = sum_b |act[b,v]|
__global__ void flops_reg_column_sum_kernel(
    const float* __restrict__ activations,
    float* __restrict__ col_sums,
    int batch_size,
    int vocab_size
) {
    int vocab_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vocab_idx >= vocab_size) return;

    float sum = 0.0f;
    #pragma unroll 8
    for (int b = 0; b < batch_size; b++) {
        sum += fabsf(activations[b * vocab_size + vocab_idx]);
    }
    col_sums[vocab_idx] = sum;
}

// loss = sum_v (col_sums[v] / B)^2
__global__ void flops_reg_final_kernel(
    const float* __restrict__ col_sums,
    float* __restrict__ loss,
    int vocab_size,
    float batch_size_inv
) {
    __shared__ float shared_sum[32];
    float local_sum = 0.0f;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab_size; v += gridDim.x * blockDim.x) {
        float mean = col_sums[v] * batch_size_inv;
        local_sum += mean * mean;
    }

    float block_sum = block_reduce_sum(local_sum, shared_sum);
    if (threadIdx.x == 0) atomicAdd(loss, block_sum);
}

// Fused for batch_size <= 64
__global__ void flops_reg_fused_kernel(
    const float* __restrict__ activations,
    float* __restrict__ loss,
    int batch_size,
    int vocab_size
) {
    __shared__ float shared_sum[32];
    float local_sum = 0.0f;
    float batch_inv = 1.0f / batch_size;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab_size; v += gridDim.x * blockDim.x) {
        float col_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            col_sum += fabsf(activations[b * vocab_size + v]);
        }
        float mean = col_sum * batch_inv;
        local_sum += mean * mean;
    }

    float block_sum = block_reduce_sum(local_sum, shared_sum);
    if (threadIdx.x == 0) atomicAdd(loss, block_sum);
}

// DF-weighted col_sums[v] = df_weights[v] * sum_b |act[b,v]|
__global__ void df_flops_reg_column_sum_kernel(
    const float* __restrict__ activations,
    const float* __restrict__ df_weights,
    float* __restrict__ col_sums,
    int batch_size,
    int vocab_size
) {
    int vocab_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vocab_idx >= vocab_size) return;

    float sum = 0.0f;
    #pragma unroll 8
    for (int b = 0; b < batch_size; b++) {
        sum += fabsf(activations[b * vocab_size + vocab_idx]);
    }
    // Apply DF weighting
    col_sums[vocab_idx] = sum * df_weights[vocab_idx];
}

// DF-weighted fused for batch_size <= 64
__global__ void df_flops_reg_fused_kernel(
    const float* __restrict__ activations,
    const float* __restrict__ df_weights,
    float* __restrict__ loss,
    int batch_size,
    int vocab_size
) {
    __shared__ float shared_sum[32];
    float local_sum = 0.0f;
    float batch_inv = 1.0f / batch_size;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab_size; v += gridDim.x * blockDim.x) {
        float col_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            col_sum += fabsf(activations[b * vocab_size + v]);
        }
        float mean = col_sum * batch_inv;
        float df_w = df_weights[v];
        local_sum += df_w * mean * mean;  // DF weighting
    }

    float block_sum = block_reduce_sum(local_sum, shared_sum);
    if (threadIdx.x == 0) atomicAdd(loss, block_sum);
}

void launch_splade_aggregate(
    const float* logits, const float* attention_mask, float* output,
    int batch_size, int seq_len, int vocab_size, cudaStream_t stream
) {
    if (vocab_size % 4 == 0) {
        int vocab_size_div4 = vocab_size / 4;
        int threads = BLOCK_SIZE_AGGREGATE;
        dim3 grid(batch_size, (vocab_size_div4 + threads - 1) / threads);

        splade_aggregate_kernel_vectorized<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(logits),
            attention_mask,
            reinterpret_cast<float4*>(output),
            batch_size, seq_len, vocab_size_div4
        );
    } else {
        int threads = BLOCK_SIZE_AGGREGATE;
        dim3 grid(batch_size, (vocab_size + threads - 1) / threads);

        splade_aggregate_kernel<<<grid, threads, 0, stream>>>(
            logits, attention_mask, output,
            batch_size, seq_len, vocab_size
        );
    }
}

void launch_flops_reg(
    const float* activations, float* loss, float* workspace,
    int batch_size, int vocab_size, cudaStream_t stream
) {
    cudaMemsetAsync(loss, 0, sizeof(float), stream);
    int threads = BLOCK_SIZE_FLOPS;

    if (batch_size <= 64) {
        int blocks = min((vocab_size + threads - 1) / threads, 256);
        flops_reg_fused_kernel<<<blocks, threads, 0, stream>>>(activations, loss, batch_size, vocab_size);
    } else {
        int blocks1 = (vocab_size + threads - 1) / threads;
        flops_reg_column_sum_kernel<<<blocks1, threads, 0, stream>>>(activations, workspace, batch_size, vocab_size);
        int blocks2 = min((vocab_size + threads - 1) / threads, 256);
        flops_reg_final_kernel<<<blocks2, threads, 0, stream>>>(workspace, loss, vocab_size, 1.0f / batch_size);
    }
}

void launch_df_flops_reg(
    const float* activations, const float* df_weights, float* loss, float* workspace,
    int batch_size, int vocab_size, cudaStream_t stream
) {
    cudaMemsetAsync(loss, 0, sizeof(float), stream);
    int threads = BLOCK_SIZE_FLOPS;

    if (batch_size <= 64) {
        int blocks = min((vocab_size + threads - 1) / threads, 256);
        df_flops_reg_fused_kernel<<<blocks, threads, 0, stream>>>(activations, df_weights, loss, batch_size, vocab_size);
    } else {
        int blocks1 = (vocab_size + threads - 1) / threads;
        df_flops_reg_column_sum_kernel<<<blocks1, threads, 0, stream>>>(activations, df_weights, workspace, batch_size, vocab_size);
        int blocks2 = min((vocab_size + threads - 1) / threads, 256);
        flops_reg_final_kernel<<<blocks2, threads, 0, stream>>>(workspace, loss, vocab_size, 1.0f / batch_size);
    }
}

size_t get_flops_reg_workspace_size(int vocab_size) {
    return vocab_size * sizeof(float);
}
