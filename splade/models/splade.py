import torch
import torch.nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from splade.models.layers.activation import DReLU


def _get_nested_attr(obj, path: str):
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


class SpladeModel(torch.nn.Module):

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.vocab_size = config.vocab_size
        self.padded_vocab_size = (self.vocab_size + 7) // 8 * 8

        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")

        self.vocab_transform = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_projector = torch.nn.Linear(config.hidden_size, self.padded_vocab_size)
        self.vocab_layer_norm = torch.nn.LayerNorm(config.hidden_size)

        self.activation = DReLU(self.padded_vocab_size)

        if self.padded_vocab_size > self.vocab_size:
            with torch.no_grad():
                self.vocab_projector.weight[self.vocab_size:].zero_()
                self.vocab_projector.bias[self.vocab_size:].zero_()

        masked_lm = AutoModelForMaskedLM.from_pretrained(model_name)
        if "distilbert" in model_name:
            transform_path, norm_path, proj_path = (
                "vocab_transform",
                "vocab_layer_norm",
                "vocab_projector",
            )
        else:
            transform_path, norm_path, proj_path = (
                "cls.predictions.transform.dense",
                "cls.predictions.transform.LayerNorm",
                "cls.predictions.decoder",
            )
        self.vocab_transform.load_state_dict(
            _get_nested_attr(masked_lm, transform_path).state_dict()
        )
        self.vocab_layer_norm.load_state_dict(
            _get_nested_attr(masked_lm, norm_path).state_dict()
        )
        pretrained_proj = _get_nested_attr(masked_lm, proj_path)
        with torch.no_grad():
            self.vocab_projector.weight[:self.vocab_size].copy_(pretrained_proj.weight)
            self.vocab_projector.bias[:self.vocab_size].copy_(pretrained_proj.bias)
        del masked_lm

        self.classifier = torch.nn.Linear(self.padded_vocab_size, num_labels)

    def _splade_head(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = self.vocab_transform(hidden)
        transformed = torch.nn.functional.gelu(transformed)
        transformed = self.vocab_layer_norm(transformed)
        mlm_logits = self.vocab_projector(transformed)

        activated = self.activation(mlm_logits)

        log_activations = torch.log1p(activated)
        masked_activations = log_activations.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 0.0
        )
        sparse_vector = masked_activations.max(dim=1).values

        return self.classifier(sparse_vector), sparse_vector

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self._splade_head(hidden, attention_mask)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.bert.embeddings(input_ids)

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(
            inputs_embeds=embeddings, attention_mask=attention_mask
        ).last_hidden_state
        return self._splade_head(hidden, attention_mask)
