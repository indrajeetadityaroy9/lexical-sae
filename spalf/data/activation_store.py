"""Activation streaming via HuggingFace model hooks."""

from collections.abc import Iterator
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda")


class ActivationStore:
    """Streams residual activations from HuggingFace causal language models."""

    def __init__(
        self,
        model_name: str,
        hook_point: str,
        dataset_name: str,
        batch_size: int,
        seq_len: int = 128,
        text_column: str = "text",
        dataset_split: str = "train",
        dataset_config: str = "",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.hook_point = hook_point
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.text_column = text_column
        self.dataset_split = dataset_split
        self.dataset_config = dataset_config
        self.seed = seed
        self.device = device

        self._hook_handle = None
        self._captured_activations: torch.Tensor | None = None
        self._token_iter: Iterator[torch.Tensor] | None = None

        self.model = self._load_model()
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self) -> AutoModelForCausalLM:
        """Load HuggingFace causal language model."""
        print(
            json.dumps(
                {
                    "event": "model_loaded",
                    "backend": "huggingface",
                    "model_name": self.model_name,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self._hf_target_module = self._resolve_hf_module(model)
        self._register_hf_hook(model)
        return model

    def _register_hf_hook(self, model: AutoModelForCausalLM) -> None:
        """Register a forward hook to capture residual stream activations."""

        def hook_fn(_module, _input, output):
            self._captured_activations = output[0].detach()

        self._hook_handle = self._hf_target_module.register_forward_hook(hook_fn)

    def _parse_layer_index(self) -> int:
        """Extract layer index from hook_point string like 'gpt_neox.layers.6'."""
        for part in self.hook_point.split("."):
            if part.isdigit():
                return int(part)
        raise ValueError(f"Unable to parse layer index from hook point: {self.hook_point}")

    def _resolve_hf_module(self, model: AutoModelForCausalLM):
        """Resolve the HuggingFace module to hook via named_modules() lookup."""
        modules = dict(model.named_modules())

        if self.hook_point in modules:
            return modules[self.hook_point]

        layer_idx = self._parse_layer_index()
        for pattern in [
            f"model.layers.{layer_idx}",
            f"gpt_neox.layers.{layer_idx}",
            f"transformer.h.{layer_idx}",
            f"model.decoder.layers.{layer_idx}",
            f"model.model.layers.{layer_idx}",
        ]:
            if pattern in modules:
                return modules[pattern]

        layer_modules = [n for n, _ in model.named_modules() if n and any(c.isdigit() for c in n)]
        raise ValueError(
            f"Cannot resolve hook target for '{self.hook_point}'. "
            f"Set hook_point to a PyTorch module path. "
            f"Layer modules: {layer_modules}"
        )

    def _token_generator(self) -> Iterator[torch.Tensor]:
        """Yield batches of token IDs. Shuffled streaming with batched tokenization."""
        target_len = self.batch_size * self.seq_len
        tokenize_batch_size = 256
        shuffle_buffer_size = 10_000

        epoch = 0
        while True:
            load_kwargs = {"split": self.dataset_split, "streaming": True}
            if self.dataset_config:
                load_kwargs["name"] = self.dataset_config
            dataset = load_dataset(self.dataset_name, **load_kwargs)
            dataset = dataset.select_columns([self.text_column])
            dataset = dataset.shuffle(seed=self.seed, buffer_size=shuffle_buffer_size)
            dataset.set_epoch(epoch)

            token_buffer: list[int] = []
            text_batch: list[str] = []

            for example in dataset:
                text_batch.append(example[self.text_column])

                if len(text_batch) >= tokenize_batch_size:
                    encoded = self.tokenizer(
                        text_batch, add_special_tokens=False
                    )["input_ids"]
                    for ids in encoded:
                        token_buffer.extend(ids)
                    text_batch = []

                    while len(token_buffer) >= target_len:
                        batch_tokens = token_buffer[:target_len]
                        token_buffer = token_buffer[target_len:]
                        yield torch.tensor(batch_tokens, dtype=torch.long).reshape(
                            self.batch_size, self.seq_len
                        )

            if text_batch:
                encoded = self.tokenizer(
                    text_batch, add_special_tokens=False
                )["input_ids"]
                for ids in encoded:
                    token_buffer.extend(ids)
                while len(token_buffer) >= target_len:
                    batch_tokens = token_buffer[:target_len]
                    token_buffer = token_buffer[target_len:]
                    yield torch.tensor(batch_tokens, dtype=torch.long).reshape(
                        self.batch_size, self.seq_len
                    )

            epoch += 1

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        """Return next activation batch flattened to [N, d_model]."""
        if self._token_iter is None:
            self._token_iter = self._token_generator()

        tokens = next(self._token_iter).to(self.device)
        self.model(tokens)
        acts = self._captured_activations

        return acts.reshape(-1, acts.shape[-1]).float()

    def get_unembedding_matrix(self) -> torch.Tensor:
        """Get W_vocab: the unembedding matrix [d_model, V]."""
        lm_head = self.model.get_output_embeddings()
        return lm_head.weight.T.float()

    def get_model(self) -> AutoModelForCausalLM:
        """Return the underlying model (for KL computation via patched forward)."""
        return self.model

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size
