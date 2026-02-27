"""In-memory activation buffer with shuffling for decorrelation."""

from __future__ import annotations

import torch

from src.data.activation_store import ActivationStore


class ActivationBuffer:
    """Accumulates activations in a large buffer and serves shuffled batches.

    Eliminates sequential correlation in activation ordering by maintaining
    a buffer of size >> batch_size and sampling randomly from it.
    """

    def __init__(
        self,
        store: ActivationStore,
        buffer_size: int = 2**20,  # ~1M activations
    ) -> None:
        self.store = store
        self.buffer_size = buffer_size
        self._buffer: torch.Tensor | None = None
        self._ptr = 0  # Next index to overwrite
        self._total_tokens_served = 0

    def _fill_buffer(self) -> None:
        """Fill the buffer from the activation store."""
        chunks: list[torch.Tensor] = []
        total = 0
        while total < self.buffer_size:
            batch = self.store.next_batch()
            chunks.append(batch)
            total += batch.shape[0]
        self._buffer = torch.cat(chunks, dim=0)[: self.buffer_size]
        self._ptr = 0

    def _refill_half(self) -> None:
        """Replace half the buffer with fresh activations."""
        half = self.buffer_size // 2
        chunks: list[torch.Tensor] = []
        total = 0
        while total < half:
            batch = self.store.next_batch()
            chunks.append(batch)
            total += batch.shape[0]
        fresh = torch.cat(chunks, dim=0)[:half]

        # Write fresh data starting at _ptr, wrapping around
        end = self._ptr + half
        if end <= self.buffer_size:
            self._buffer[self._ptr : end] = fresh
        else:
            first_part = self.buffer_size - self._ptr
            self._buffer[self._ptr :] = fresh[:first_part]
            self._buffer[: half - first_part] = fresh[first_part:]
        self._ptr = end % self.buffer_size

    def next_batch(self, batch_size: int) -> torch.Tensor:
        """Return a shuffled batch of activations [batch_size, d_model]."""
        if self._buffer is None:
            self._fill_buffer()

        # Random sample from buffer (with replacement if batch > buffer, shouldn't happen)
        indices = torch.randint(0, self.buffer_size, (batch_size,))
        batch = self._buffer[indices].clone()
        self._total_tokens_served += batch_size

        # Periodically refill half the buffer to keep it fresh
        if self._total_tokens_served % self.buffer_size < batch_size:
            self._refill_half()

        return batch
