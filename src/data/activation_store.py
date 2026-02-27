"""Hybrid activation streaming: TransformerLens for supported models, HuggingFace hooks otherwise."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ModelType = HookedTransformer | AutoModelForCausalLM


class ActivationStore:
    """Streams residual-stream activations from a transformer model.

    Hybrid approach:
    - Uses TransformerLens HookedTransformer for supported models (Pythia, GPT-2).
    - Uses manual PyTorch forward hooks for unsupported models (Llama-3-8B).
    """

    def __init__(
        self,
        model_name: str,
        hook_point: str,
        dataset_name: str,
        batch_size: int,
        seq_len: int = 128,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.model_name = model_name
        self.hook_point = hook_point
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        self._use_transformerlens = False
        self._hook_handle = None
        self._captured_activations: torch.Tensor | None = None
        self._token_iter: Iterator[torch.Tensor] | None = None

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Load tokenizer
        if self._use_transformerlens:
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self) -> ModelType:
        """Load via TransformerLens if the model is supported, otherwise HuggingFace."""
        try:
            model = HookedTransformer.from_pretrained(
                self.model_name,
                device=str(self.device),
                dtype=torch.bfloat16,
            )
            self._use_transformerlens = True
            print(f"Loaded {self.model_name} via TransformerLens")
            return model
        except ValueError:
            # TransformerLens raises ValueError for unsupported model architectures
            pass

        print(f"TransformerLens does not support {self.model_name}, using HuggingFace hooks")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self._register_hf_hook(model)
        return model

    def _register_hf_hook(self, model: AutoModelForCausalLM) -> None:
        """Register a forward hook to capture residual stream activations."""
        layer_idx = self._parse_layer_index()
        target_module = self._resolve_hf_layer(model, layer_idx)

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._captured_activations = output[0].detach()
            else:
                self._captured_activations = output.detach()

        self._hook_handle = target_module.register_forward_hook(hook_fn)
        print(f"Registered HF hook on layer {layer_idx}")

    def _parse_layer_index(self) -> int:
        """Extract layer index from hook_point string like 'blocks.6.hook_resid_post'."""
        parts = self.hook_point.split(".")
        for part in parts:
            if part.isdigit():
                return int(part)

    def _resolve_hf_layer(self, model: AutoModelForCausalLM, layer_idx: int):
        """Resolve the transformer layer module for HuggingFace models."""
        for attr_path in [
            f"model.layers.{layer_idx}",  # Llama, Mistral
            f"gpt_neox.layers.{layer_idx}",  # Pythia
            f"transformer.h.{layer_idx}",  # GPT-2
            f"model.model.layers.{layer_idx}",  # Some wrapped models
        ]:
            module = model
            resolved = True
            for attr in attr_path.split("."):
                if attr.isdigit():
                    try:
                        module = module[int(attr)]
                    except (IndexError, TypeError):
                        resolved = False
                        break
                else:
                    if not hasattr(module, attr):
                        resolved = False
                        break
                    module = getattr(module, attr)
            if resolved:
                print(f"Resolved HF layer via {attr_path}")
                return module

    def _token_generator(self) -> Iterator[torch.Tensor]:
        """Yield batches of token IDs from the dataset. Cycles indefinitely."""
        target_len = self.batch_size * self.seq_len

        while True:
            dataset = load_dataset(self.dataset_name, split="train", streaming=True)
            buffer: list[int] = []

            for example in dataset:
                text = example.get("text", "")
                if not text:
                    continue

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(tokens)

                while len(buffer) >= target_len:
                    batch_tokens = buffer[:target_len]
                    buffer = buffer[target_len:]
                    tensor = torch.tensor(batch_tokens, dtype=torch.long)
                    yield tensor.reshape(self.batch_size, self.seq_len)

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        """Get next batch of activations. Returns [N, d_model] flattened."""
        if self._token_iter is None:
            self._token_iter = self._token_generator()

        tokens = next(self._token_iter).to(self.device)  # [batch, seq_len]

        if self._use_transformerlens:
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=self.hook_point,
                stop_at_layer=self._parse_layer_index() + 1,
            )
            acts = cache[self.hook_point]  # [batch, seq_len, d_model]
        else:
            self.model(tokens)
            acts = self._captured_activations  # [batch, seq_len, d_model]

        # Flatten to [batch * seq_len, d_model]
        return acts.reshape(-1, acts.shape[-1]).float()

    def get_unembedding_matrix(self) -> torch.Tensor:
        """Get W_vocab: the unembedding matrix [d_model, V]."""
        if self._use_transformerlens:
            return self.model.W_U.float()
        else:
            lm_head = self.model.get_output_embeddings()
            return lm_head.weight.T.float()  # [d_model, V]

    def get_model(self) -> ModelType:
        """Return the underlying model (for Phase 2 KL computation)."""
        return self.model

    @property
    def d_model(self) -> int:
        if self._use_transformerlens:
            return self.model.cfg.d_model
        else:
            return self.model.config.hidden_size

