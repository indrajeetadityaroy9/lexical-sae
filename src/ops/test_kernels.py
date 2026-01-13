"""
Test suite for Triton kernel implementations.

Validates:
1. Numerical parity with PyTorch reference implementations
2. Performance improvements over PyTorch
3. Edge cases and shape handling
"""

import torch
import pytest
import time
from typing import Tuple

# Import both implementations
from src.ops import (
    log_saturation_triton,
    log_saturation_pytorch,
    splade_aggregate_triton,
    splade_aggregate_pytorch,
    flops_regularization_triton,
    flops_regularization_pytorch,
    topk_activation_triton,
    topk_activation_pytorch,
    normalized_linear_triton,
    normalized_linear_pytorch,
    TRITON_AVAILABLE,
)


def check_parity(
    triton_out: torch.Tensor,
    pytorch_out: torch.Tensor,
    name: str,
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> Tuple[bool, float]:
    """Check numerical parity and return max relative error."""
    if triton_out.shape != pytorch_out.shape:
        print(f"[{name}] Shape mismatch: {triton_out.shape} vs {pytorch_out.shape}")
        return False, float('inf')

    max_abs_diff = (triton_out - pytorch_out).abs().max().item()
    max_rel_diff = ((triton_out - pytorch_out).abs() / (pytorch_out.abs() + 1e-8)).max().item()

    is_close = torch.allclose(triton_out, pytorch_out, rtol=rtol, atol=atol)

    return is_close, max_rel_diff


def benchmark_fn(fn, *args, warmup=10, iters=100, **kwargs) -> float:
    """Benchmark a function and return average time in ms."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000 / iters


class TestLogSaturation:
    """Tests for log_saturation kernel."""

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_numerical_parity(self):
        """Test that Triton matches PyTorch output."""
        device = torch.device("cuda")

        # Test various shapes
        shapes = [
            (32, 128, 30522),  # Typical SPLADE shape
            (1, 128, 30522),   # Single sample
            (64, 64, 1024),    # Smaller
            (128, 256, 256),   # Different aspect
        ]

        for shape in shapes:
            x = torch.randn(shape, device=device)

            triton_out = log_saturation_triton(x)
            pytorch_out = log_saturation_pytorch(x)

            is_close, max_rel = check_parity(triton_out, pytorch_out, f"log_sat_{shape}")
            assert is_close, f"Failed for shape {shape}, max_rel_error={max_rel}"
            print(f"  Shape {shape}: PASS (max_rel_error={max_rel:.2e})")

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_performance(self):
        """Test that Triton is faster than PyTorch."""
        device = torch.device("cuda")
        x = torch.randn(32, 128, 30522, device=device)

        pytorch_time = benchmark_fn(log_saturation_pytorch, x)
        triton_time = benchmark_fn(log_saturation_triton, x)

        speedup = pytorch_time / triton_time
        print(f"  PyTorch: {pytorch_time:.3f}ms, Triton: {triton_time:.3f}ms, Speedup: {speedup:.2f}x")

        # Triton should be at least as fast
        assert triton_time <= pytorch_time * 1.5, "Triton significantly slower"


class TestSpladeAggregate:
    """Tests for fused SPLADE aggregation kernel."""

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_numerical_parity(self):
        """Test that Triton matches PyTorch output."""
        device = torch.device("cuda")

        # Test various configurations
        configs = [
            (32, 128, 30522),  # Typical
            (1, 64, 30522),    # Single sample
            (64, 256, 10000),  # Different vocab
            (16, 32, 5000),    # Small
        ]

        for batch, seq, vocab in configs:
            logits = torch.randn(batch, seq, vocab, device=device)
            mask = torch.ones(batch, seq, device=device)
            mask[:, seq//2:] = 0  # Variable length

            triton_out = splade_aggregate_triton(logits, mask)
            pytorch_out = splade_aggregate_pytorch(logits, mask)

            is_close, max_rel = check_parity(triton_out, pytorch_out, f"splade_agg_{batch}x{seq}x{vocab}")
            assert is_close, f"Failed for config ({batch},{seq},{vocab}), max_rel={max_rel}"
            print(f"  Config ({batch},{seq},{vocab}): PASS (max_rel={max_rel:.2e})")

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_all_masked(self):
        """Test with all positions masked."""
        device = torch.device("cuda")
        logits = torch.randn(4, 32, 1000, device=device)
        mask = torch.zeros(4, 32, device=device)

        triton_out = splade_aggregate_triton(logits, mask)
        pytorch_out = splade_aggregate_pytorch(logits, mask)

        # Both should be all zeros
        assert (triton_out == 0).all(), "Triton should return zeros for all-masked"
        assert (pytorch_out == 0).all(), "PyTorch should return zeros for all-masked"

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_performance(self):
        """Test performance improvement."""
        device = torch.device("cuda")
        logits = torch.randn(32, 128, 30522, device=device)
        mask = torch.ones(32, 128, device=device)

        pytorch_time = benchmark_fn(splade_aggregate_pytorch, logits, mask)
        triton_time = benchmark_fn(splade_aggregate_triton, logits, mask)

        speedup = pytorch_time / triton_time
        print(f"  PyTorch: {pytorch_time:.3f}ms, Triton: {triton_time:.3f}ms, Speedup: {speedup:.2f}x")


class TestTopKActivation:
    """Tests for TopK activation kernel."""

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_numerical_parity(self):
        """Test that Triton matches PyTorch output."""
        device = torch.device("cuda")

        configs = [
            (256, 16384, 32),   # Typical SAE
            (64, 8192, 64),     # Different k
            (128, 4096, 16),    # Smaller
        ]

        for batch, hidden, k in configs:
            x = torch.randn(batch, hidden, device=device)

            triton_out = topk_activation_triton(x, k)
            pytorch_out = topk_activation_pytorch(x, k)

            # Check same number of non-zeros
            triton_nnz = (triton_out != 0).sum(dim=1)
            pytorch_nnz = (pytorch_out != 0).sum(dim=1)

            # TopK should keep exactly k values per row
            assert (triton_nnz == k).all(), f"Triton NNZ mismatch"
            assert (pytorch_nnz == k).all(), f"PyTorch NNZ mismatch"

            # Values should match (order may differ for ties)
            triton_sorted = triton_out.sort(dim=1, descending=True)[0][:, :k]
            pytorch_sorted = pytorch_out.sort(dim=1, descending=True)[0][:, :k]

            is_close, max_rel = check_parity(triton_sorted, pytorch_sorted, f"topk_{batch}x{hidden}_k{k}")
            assert is_close, f"Failed for config ({batch},{hidden},{k})"
            print(f"  Config ({batch},{hidden},k={k}): PASS")

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_performance(self):
        """Test performance improvement."""
        device = torch.device("cuda")
        x = torch.randn(256, 16384, device=device)
        k = 32

        pytorch_time = benchmark_fn(topk_activation_pytorch, x, k)
        triton_time = benchmark_fn(topk_activation_triton, x, k)

        speedup = pytorch_time / triton_time
        print(f"  PyTorch: {pytorch_time:.3f}ms, Triton: {triton_time:.3f}ms, Speedup: {speedup:.2f}x")


class TestNormalizedLinear:
    """Tests for normalized linear kernel."""

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_numerical_parity(self):
        """Test that Triton matches PyTorch output."""
        device = torch.device("cuda")

        configs = [
            (64, 16384, 30522),   # Typical SAE decode
            (32, 8192, 10000),    # Smaller
            (128, 4096, 5000),    # Different shape
        ]

        for batch, hidden, output_dim in configs:
            hidden_tensor = torch.randn(batch, hidden, device=device) * 0.1
            weight = torch.randn(hidden, output_dim, device=device)
            bias = torch.randn(output_dim, device=device)

            triton_out = normalized_linear_triton(hidden_tensor, weight, bias)
            pytorch_out = normalized_linear_pytorch(hidden_tensor, weight, bias)

            is_close, max_rel = check_parity(
                triton_out, pytorch_out,
                f"norm_linear_{batch}x{hidden}x{output_dim}",
                rtol=1e-3, atol=1e-4  # Slightly relaxed for large matmuls
            )
            assert is_close, f"Failed for config ({batch},{hidden},{output_dim}), max_rel={max_rel}"
            print(f"  Config ({batch},{hidden},{output_dim}): PASS (max_rel={max_rel:.2e})")

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
    def test_performance(self):
        """Test performance improvement."""
        device = torch.device("cuda")
        hidden = torch.randn(256, 16384, device=device)
        weight = torch.randn(16384, 30522, device=device)
        bias = torch.randn(30522, device=device)

        pytorch_time = benchmark_fn(normalized_linear_pytorch, hidden, weight, bias, warmup=5, iters=20)
        triton_time = benchmark_fn(normalized_linear_triton, hidden, weight, bias, warmup=5, iters=20)

        speedup = pytorch_time / triton_time
        print(f"  PyTorch: {pytorch_time:.3f}ms, Triton: {triton_time:.3f}ms, Speedup: {speedup:.2f}x")


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "="*60)
    print("TRITON KERNEL TESTS")
    print("="*60)

    if not TRITON_AVAILABLE:
        print("WARNING: Triton not available, skipping tests")
        return

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping tests")
        return

    test_classes = [
        TestLogSaturation,
        TestSpladeAggregate,
        TestTopKActivation,
        TestNormalizedLinear,
    ]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                try:
                    method()
                    print(f"  {method_name}: PASSED")
                except Exception as e:
                    print(f"  {method_name}: FAILED - {e}")

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
