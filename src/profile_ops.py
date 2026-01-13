"""
Profile computational bottlenecks in the SPLADE pipeline.

This script identifies hot paths by measuring:
- GPU kernel time per operation
- Memory bandwidth utilization
- Kernel launch overhead
- Synchronization points
"""

import torch
import torch.nn.functional as F
import time
import argparse
from contextlib import contextmanager
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA required for profiling"


@dataclass
class ProfileResult:
    """Container for profiling results."""
    name: str
    gpu_time_ms: float
    cpu_time_ms: float
    memory_mb: float
    throughput_gflops: float = 0.0


class OperationProfiler:
    """Profile individual operations with CUDA events."""

    def __init__(self, warmup_iters: int = 10, profile_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.results: List[ProfileResult] = []

    @contextmanager
    def profile_op(self, name: str):
        """Context manager for profiling a single operation."""
        # Synchronize before starting
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_cpu = time.perf_counter()
        start_event.record()

        yield

        end_event.record()
        torch.cuda.synchronize()
        end_cpu = time.perf_counter()

        gpu_time_ms = start_event.elapsed_time(end_event)
        cpu_time_ms = (end_cpu - start_cpu) * 1000

        # Get memory usage
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        self.results.append(ProfileResult(
            name=name,
            gpu_time_ms=gpu_time_ms,
            cpu_time_ms=cpu_time_ms,
            memory_mb=memory_mb
        ))

    def benchmark_op(self, name: str, op_fn, *args, **kwargs) -> ProfileResult:
        """Benchmark an operation over multiple iterations."""
        # Warmup
        for _ in range(self.warmup_iters):
            op_fn(*args, **kwargs)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_cpu = time.perf_counter()
        start_event.record()

        for _ in range(self.profile_iters):
            op_fn(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        end_cpu = time.perf_counter()

        gpu_time_ms = start_event.elapsed_time(end_event) / self.profile_iters
        cpu_time_ms = (end_cpu - start_cpu) * 1000 / self.profile_iters
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        result = ProfileResult(
            name=name,
            gpu_time_ms=gpu_time_ms,
            cpu_time_ms=cpu_time_ms,
            memory_mb=memory_mb
        )
        self.results.append(result)
        return result


def profile_splade_ops(batch_size: int = 32, seq_len: int = 128, vocab_size: int = 30522):
    """Profile SPLADE-specific operations."""
    device = torch.device("cuda")
    profiler = OperationProfiler(warmup_iters=20, profile_iters=100)

    print(f"\n{'='*60}")
    print(f"SPLADE Operations Profiling")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Vocab: {vocab_size}")
    print(f"{'='*60}\n")

    # Create test tensors
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    attention_mask[:, seq_len//2:] = 0  # Simulate variable length
    sparse_vec = torch.randn(batch_size, vocab_size, device=device).abs() * 0.1
    sparse_vec[sparse_vec < 0.05] = 0  # Make sparse

    # 1. Log-saturation activation: log(1 + ReLU(x))
    def log_saturation():
        return torch.log1p(F.relu(logits))

    result = profiler.benchmark_op("log_saturation", log_saturation)
    print(f"1. Log-Saturation (log1p + relu):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 2. Masked expansion + element-wise multiply
    def masked_multiply():
        mask = attention_mask.unsqueeze(-1).expand(batch_size, seq_len, vocab_size)
        weights = torch.log1p(F.relu(logits))
        return weights * mask

    result = profiler.benchmark_op("masked_multiply", masked_multiply)
    print(f"2. Masked Multiply (expand + mul):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 3. Max pooling over sequence
    weights = torch.randn(batch_size, seq_len, vocab_size, device=device).abs()

    def max_pool_seq():
        return torch.max(weights, dim=1)[0]

    result = profiler.benchmark_op("max_pool_sequence", max_pool_seq)
    print(f"3. Max Pool (over seq_len={seq_len}):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 4. Full SPLADE aggregation (fused)
    def splade_aggregation():
        weights = torch.log1p(F.relu(logits))
        mask = attention_mask.unsqueeze(-1).expand_as(weights)
        weights = weights * mask
        doc_vec, _ = torch.max(weights, dim=1)
        return doc_vec

    result = profiler.benchmark_op("splade_full_aggregation", splade_aggregation)
    print(f"4. Full SPLADE Aggregation (log1p+relu+mask+max):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 5. FLOPS regularization
    def flops_reg():
        mean_act = torch.mean(torch.abs(sparse_vec), dim=0)
        return torch.sum(mean_act ** 2)

    result = profiler.benchmark_op("flops_regularization", flops_reg)
    print(f"5. FLOPS Regularization:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 6. Linear classifier (vocab_size -> 1)
    classifier = torch.nn.Linear(vocab_size, 1).to(device)

    def linear_classify():
        return classifier(sparse_vec)

    result = profiler.benchmark_op("linear_classifier", linear_classify)
    print(f"6. Linear Classifier ({vocab_size} -> 1):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    return profiler.results


def profile_sae_ops(batch_size: int = 256, input_dim: int = 30522, hidden_dim: int = 16384, k: int = 32):
    """Profile SAE-specific operations."""
    device = torch.device("cuda")
    profiler = OperationProfiler(warmup_iters=20, profile_iters=100)

    print(f"\n{'='*60}")
    print(f"SAE Operations Profiling")
    print(f"Batch: {batch_size}, Input: {input_dim}, Hidden: {hidden_dim}, K: {k}")
    print(f"{'='*60}\n")

    # Create test tensors
    x = torch.randn(batch_size, input_dim, device=device)
    x[x.abs() < 0.5] = 0  # Sparse input
    hidden = torch.randn(batch_size, hidden_dim, device=device)

    # 1. Encoder linear
    encoder = torch.nn.Linear(input_dim, hidden_dim).to(device)

    def encoder_forward():
        return encoder(x)

    result = profiler.benchmark_op("sae_encoder", encoder_forward)
    print(f"1. SAE Encoder ({input_dim} -> {hidden_dim}):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 2. TopK activation (current scatter-based)
    def topk_scatter():
        topk_values, topk_indices = torch.topk(hidden, k, dim=-1)
        sparse_output = torch.zeros_like(hidden)
        sparse_output.scatter_(-1, topk_indices, topk_values)
        return sparse_output

    result = profiler.benchmark_op("topk_scatter", topk_scatter)
    print(f"2. TopK Scatter (k={k}):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 3. TopK with masking (alternative)
    def topk_mask():
        topk_values, topk_indices = torch.topk(hidden, k, dim=-1)
        # Get threshold (k-th largest value per row)
        threshold = topk_values[:, -1:].expand_as(hidden)
        mask = hidden >= threshold
        return hidden * mask

    result = profiler.benchmark_op("topk_mask", topk_mask)
    print(f"3. TopK Mask (alternative):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 4. Decoder linear
    decoder = torch.nn.Linear(hidden_dim, input_dim).to(device)
    sparse_hidden = torch.zeros(batch_size, hidden_dim, device=device)
    indices = torch.randint(0, hidden_dim, (batch_size, k), device=device)
    values = torch.randn(batch_size, k, device=device)
    sparse_hidden.scatter_(-1, indices, values)

    def decoder_forward():
        return decoder(sparse_hidden)

    result = profiler.benchmark_op("sae_decoder", decoder_forward)
    print(f"4. SAE Decoder ({hidden_dim} -> {input_dim}):")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 5. Tied-weight decode with normalization
    def tied_decode_normalized():
        weights = encoder.weight.t()
        weights_norm = F.normalize(weights, dim=1)
        return F.linear(sparse_hidden, weights_norm)

    result = profiler.benchmark_op("tied_decode_norm", tied_decode_normalized)
    print(f"5. Tied Decode + Normalize:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 6. MSE loss
    reconstruction = torch.randn(batch_size, input_dim, device=device)

    def mse_loss():
        return F.mse_loss(reconstruction, x)

    result = profiler.benchmark_op("mse_loss", mse_loss)
    print(f"6. MSE Loss:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # 7. L1 sparsity on hidden
    def l1_sparsity():
        return sparse_hidden.abs().mean()

    result = profiler.benchmark_op("l1_sparsity", l1_sparsity)
    print(f"7. L1 Sparsity:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    return profiler.results


def profile_end_to_end(batch_size: int = 32, seq_len: int = 128):
    """Profile end-to-end training iteration."""
    from src.models import DistilBERTSparseClassifier
    from src.regularizers import flops_regularization

    device = torch.device("cuda")
    profiler = OperationProfiler(warmup_iters=5, profile_iters=20)

    print(f"\n{'='*60}")
    print(f"End-to-End Training Iteration Profiling")
    print(f"Batch: {batch_size}, Seq: {seq_len}")
    print(f"{'='*60}\n")

    # Create model and inputs
    model = DistilBERTSparseClassifier().to(device)
    model.train()

    input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = torch.randint(0, 2, (batch_size, 1), device=device).float()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Profile forward pass
    def forward_pass():
        return model(input_ids, attention_mask)

    result = profiler.benchmark_op("forward_pass", forward_pass)
    print(f"1. Forward Pass:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # Profile loss computation
    logits, sparse_vec = model(input_ids, attention_mask)

    def loss_computation():
        bce = criterion(logits, labels)
        flops = 1e-4 * flops_regularization(sparse_vec)
        return bce + flops

    result = profiler.benchmark_op("loss_computation", loss_computation)
    print(f"2. Loss Computation:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # Profile backward pass
    def backward_pass():
        optimizer.zero_grad()
        logits, sparse_vec = model(input_ids, attention_mask)
        loss = criterion(logits, labels) + 1e-4 * flops_regularization(sparse_vec)
        loss.backward()
        return loss

    result = profiler.benchmark_op("backward_pass", backward_pass)
    print(f"3. Full Backward Pass:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # Profile optimizer step
    def optimizer_step():
        optimizer.step()

    result = profiler.benchmark_op("optimizer_step", optimizer_step)
    print(f"4. Optimizer Step:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    # Full training iteration
    def full_iteration():
        optimizer.zero_grad()
        logits, sparse_vec = model(input_ids, attention_mask)
        loss = criterion(logits, labels) + 1e-4 * flops_regularization(sparse_vec)
        loss.backward()
        optimizer.step()
        return loss

    result = profiler.benchmark_op("full_iteration", full_iteration)
    print(f"5. Full Training Iteration:")
    print(f"   GPU: {result.gpu_time_ms:.3f}ms, CPU: {result.cpu_time_ms:.3f}ms")

    return profiler.results


def print_summary(all_results: List[ProfileResult]):
    """Print summary of profiling results."""
    print(f"\n{'='*60}")
    print("PROFILING SUMMARY (sorted by GPU time)")
    print(f"{'='*60}")

    sorted_results = sorted(all_results, key=lambda x: x.gpu_time_ms, reverse=True)

    total_time = sum(r.gpu_time_ms for r in sorted_results)

    print(f"\n{'Operation':<35} {'GPU (ms)':<12} {'% of Total':<12} {'Memory (MB)':<12}")
    print("-" * 71)

    for result in sorted_results:
        pct = (result.gpu_time_ms / total_time * 100) if total_time > 0 else 0
        print(f"{result.name:<35} {result.gpu_time_ms:<12.3f} {pct:<12.1f} {result.memory_mb:<12.1f}")

    print("-" * 71)
    print(f"{'TOTAL':<35} {total_time:<12.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--profile", type=str, default="all",
                       choices=["splade", "sae", "e2e", "all"])
    args = parser.parse_args()

    all_results = []

    if args.profile in ["splade", "all"]:
        results = profile_splade_ops(args.batch_size, args.seq_len)
        all_results.extend(results)

    if args.profile in ["sae", "all"]:
        results = profile_sae_ops(batch_size=256)
        all_results.extend(results)

    if args.profile in ["e2e", "all"]:
        results = profile_end_to_end(args.batch_size, args.seq_len)
        all_results.extend(results)

    print_summary(all_results)
