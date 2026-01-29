#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" {
    void launch_splade_aggregate(
        const float* logits,
        const float* attention_mask,
        float* output,
        int batch_size,
        int seq_len,
        int vocab_size,
        cudaStream_t stream
    );

    void launch_flops_reg(
        const float* activations,
        float* loss,
        float* workspace,
        int batch_size,
        int vocab_size,
        cudaStream_t stream
    );

    void launch_df_flops_reg(
        const float* activations,
        const float* df_weights,
        float* loss,
        float* workspace,
        int batch_size,
        int vocab_size,
        cudaStream_t stream
    );

    size_t get_flops_reg_workspace_size(int vocab_size);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// [B,S,V] -> [B,V]
torch::Tensor splade_aggregate_cuda(
    torch::Tensor logits,
    torch::Tensor attention_mask
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(attention_mask);

    TORCH_CHECK(logits.dim() == 3, "logits must be 3D [batch, seq, vocab]");
    TORCH_CHECK(attention_mask.dim() == 2, "attention_mask must be 2D [batch, seq]");

    int batch_size = logits.size(0);
    int seq_len = logits.size(1);
    int vocab_size = logits.size(2);

    TORCH_CHECK(attention_mask.size(0) == batch_size, "batch size mismatch");
    TORCH_CHECK(attention_mask.size(1) == seq_len, "seq_len mismatch");

    auto output = torch::empty({batch_size, vocab_size},
                               torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    auto mask_f32 = attention_mask.to(torch::kFloat32);

    launch_splade_aggregate(
        logits.data_ptr<float>(),
        mask_f32.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, seq_len, vocab_size,
        stream
    );

    return output;
}

// L = sum_v (mean_b |act[b,v]|)^2. [B,V] -> scalar.
torch::Tensor flops_reg_cuda(torch::Tensor activations) {
    CHECK_INPUT(activations);
    TORCH_CHECK(activations.dim() == 2, "activations must be 2D [batch, vocab]");
    TORCH_CHECK(activations.dtype() == torch::kFloat32, "activations must be float32");

    int batch_size = activations.size(0);
    int vocab_size = activations.size(1);

    auto loss = torch::zeros({1}, activations.options());

    size_t workspace_size = get_flops_reg_workspace_size(vocab_size);
    auto workspace = torch::empty({static_cast<long>(workspace_size / sizeof(float))},
                                  activations.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_flops_reg(
        activations.data_ptr<float>(),
        loss.data_ptr<float>(),
        workspace.data_ptr<float>(),
        batch_size, vocab_size,
        stream
    );

    return loss.squeeze(0);
}

// L = sum_v (df_weights[v] * mean_b |act[b,v]|)^2. [B,V], [V] -> scalar.
torch::Tensor df_flops_reg_cuda(torch::Tensor activations, torch::Tensor df_weights) {
    CHECK_INPUT(activations);
    CHECK_INPUT(df_weights);
    TORCH_CHECK(activations.dim() == 2, "activations must be 2D [batch, vocab]");
    TORCH_CHECK(df_weights.dim() == 1, "df_weights must be 1D [vocab]");
    TORCH_CHECK(activations.dtype() == torch::kFloat32, "activations must be float32");
    TORCH_CHECK(df_weights.dtype() == torch::kFloat32, "df_weights must be float32");

    int batch_size = activations.size(0);
    int vocab_size = activations.size(1);
    TORCH_CHECK(df_weights.size(0) == vocab_size, "df_weights size must match vocab_size");

    auto loss = torch::zeros({1}, activations.options());

    size_t workspace_size = get_flops_reg_workspace_size(vocab_size);
    auto workspace = torch::empty({static_cast<long>(workspace_size / sizeof(float))},
                                  activations.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_df_flops_reg(
        activations.data_ptr<float>(),
        df_weights.data_ptr<float>(),
        loss.data_ptr<float>(),
        workspace.data_ptr<float>(),
        batch_size, vocab_size,
        stream
    );

    return loss.squeeze(0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splade_aggregate", &splade_aggregate_cuda, py::arg("logits"), py::arg("attention_mask"));
    m.def("flops_reg", &flops_reg_cuda, py::arg("activations"));
    m.def("df_flops_reg", &df_flops_reg_cuda, py::arg("activations"), py::arg("df_weights"));
}
