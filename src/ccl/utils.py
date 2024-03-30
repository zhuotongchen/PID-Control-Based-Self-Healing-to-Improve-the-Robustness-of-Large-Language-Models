import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.models.distilbert.modeling_distilbert import (
    SequenceClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.opt.modeling_opt import SequenceClassifierOutputWithPast


TRANSFORMERS_MODELS_TO_EMBEDDING_MODULES_MAPPING = {
    "distilbert": {
        "control_word_embedding": True,
        "control_scheme": [
            "distilbert.transformer.layer.0",
            "distilbert.transformer.layer.1",
            "distilbert.transformer.layer.2",
            "distilbert.transformer.layer.3",
            # "distilbert.transformer.layer.4",
            # "distilbert.transformer.layer.5",
        ],
        "transformer_scheme": [
            "distilbert.transformer.layer.0",
            "distilbert.transformer.layer.1",
            "distilbert.transformer.layer.2",
            "distilbert.transformer.layer.3",
            "distilbert.transformer.layer.4",
            "distilbert.transformer.layer.5",
        ],
        "embedding_scheme": "distilbert.embeddings",
    },
    "bert": {
        "control_word_embedding": True,
        "control_scheme": [
            "bert.encoder.layer.0",
            "bert.encoder.layer.1",
            "bert.encoder.layer.2",
            # "bert.encoder.layer.3",
            # "bert.encoder.layer.4",
            # "bert.encoder.layer.5",
            # "bert.encoder.layer.6",
            # "bert.encoder.layer.7",
            # "bert.encoder.layer.8",
            # "bert.encoder.layer.9",
            # "bert.encoder.layer.10",
            # "bert.encoder.layer.11",
            # "bert.encoder.layer.12",
            # "bert.encoder.layer.13",
            # "bert.encoder.layer.14",
            # "bert.encoder.layer.15",
            # "bert.encoder.layer.16",
            # "bert.encoder.layer.17",
            # "bert.encoder.layer.18",
            # "bert.encoder.layer.19",
            # "bert.encoder.layer.20",
            # "bert.encoder.layer.21",
            # "bert.encoder.layer.22",
        ],
        "transformer_scheme": [
            "bert.encoder.layer.0",
            "bert.encoder.layer.1",
            "bert.encoder.layer.2",
            "bert.encoder.layer.3",
            "bert.encoder.layer.4",
            "bert.encoder.layer.5",
            "bert.encoder.layer.6",
            "bert.encoder.layer.7",
            "bert.encoder.layer.8",
            "bert.encoder.layer.9",
            "bert.encoder.layer.10",
            "bert.encoder.layer.11",
            "bert.encoder.layer.12",
            "bert.encoder.layer.13",
            "bert.encoder.layer.14",
            "bert.encoder.layer.15",
            "bert.encoder.layer.16",
            "bert.encoder.layer.17",
            "bert.encoder.layer.18",
            "bert.encoder.layer.19",
            "bert.encoder.layer.20",
            "bert.encoder.layer.21",
            "bert.encoder.layer.22",
            "bert.encoder.layer.23",
        ],
        "embedding_scheme": "bert.embeddings",
    },
    "roberta": {
        "control_word_embedding": True,
        "control_scheme": [
            "roberta.encoder.layer.0",
            "roberta.encoder.layer.1",
            "roberta.encoder.layer.2",
            "roberta.encoder.layer.3",
            "roberta.encoder.layer.4",
            # "roberta.encoder.layer.5",
            # "roberta.encoder.layer.6",
            # "roberta.encoder.layer.7",
            # "roberta.encoder.layer.8",
            # "roberta.encoder.layer.9",
            # "roberta.encoder.layer.10",
            # "roberta.encoder.layer.11",
            # "roberta.encoder.layer.12",
            # "roberta.encoder.layer.13",
            # "roberta.encoder.layer.14",
            # "roberta.encoder.layer.15",
            # "roberta.encoder.layer.16",
            # "roberta.encoder.layer.17",
            # "roberta.encoder.layer.18",
            # "roberta.encoder.layer.19",
            # "roberta.encoder.layer.20",
            # "roberta.encoder.layer.21",
            # "roberta.encoder.layer.22",
        ],
        "transformer_scheme": [
            "roberta.encoder.layer.0",
            "roberta.encoder.layer.1",
            "roberta.encoder.layer.2",
            "roberta.encoder.layer.3",
            "roberta.encoder.layer.4",
            "roberta.encoder.layer.5",
            "roberta.encoder.layer.6",
            "roberta.encoder.layer.7",
            "roberta.encoder.layer.8",
            "roberta.encoder.layer.9",
            "roberta.encoder.layer.10",
            "roberta.encoder.layer.11",
            "roberta.encoder.layer.12",
            "roberta.encoder.layer.13",
            "roberta.encoder.layer.14",
            "roberta.encoder.layer.15",
            "roberta.encoder.layer.16",
            "roberta.encoder.layer.17",
            "roberta.encoder.layer.18",
            "roberta.encoder.layer.19",
            "roberta.encoder.layer.20",
            "roberta.encoder.layer.21",
            "roberta.encoder.layer.22",
            "roberta.encoder.layer.23",
        ],
        "embedding_scheme": "roberta.embeddings",
    },
    "roberta-base": {
        "control_word_embedding": True,
        "control_scheme": [
            "roberta.encoder.layer.0",
            "roberta.encoder.layer.1",
            # "roberta.encoder.layer.2",
            # "roberta.encoder.layer.3",
            # "roberta.encoder.layer.4",
            # "roberta.encoder.layer.5",
            # "roberta.encoder.layer.6",
            # "roberta.encoder.layer.7",
            # "roberta.encoder.layer.8",
            # "roberta.encoder.layer.9",
            # "roberta.encoder.layer.10",
            # "roberta.encoder.layer.11",
        ],
        "transformer_scheme": [
            "roberta.encoder.layer.0",
            "roberta.encoder.layer.1",
            "roberta.encoder.layer.2",
            "roberta.encoder.layer.3",
            "roberta.encoder.layer.4",
            "roberta.encoder.layer.5",
            "roberta.encoder.layer.6",
            "roberta.encoder.layer.7",
            "roberta.encoder.layer.8",
            "roberta.encoder.layer.9",
            "roberta.encoder.layer.10",
            "roberta.encoder.layer.11",
        ],
        "embedding_scheme": "roberta.embeddings",
    },
    "opt": {
        "control_word_embedding": True,
        # "control_scheme": [
        #     # "model.decoder.layers.0",
        #     # "model.decoder.layers.1",
        #     # "model.decoder.layers.2",
        #     "model.decoder.layers.3",
        #     # "model.decoder.layers.4",
        #     # "model.decoder.layers.5",
        #     # "model.decoder.layers.6",
        #     # "model.decoder.layers.7",
        #     # "model.decoder.layers.8",
        #     # "model.decoder.layers.9",
        #     # "model.decoder.layers.10",
        #     # "model.decoder.layers.11",
        #     # "model.decoder.layers.12",
        #     # "model.decoder.layers.13",
        #     # "model.decoder.layers.14",
        #     # "model.decoder.layers.15",
        #     # "model.decoder.layers.16",
        #     # "model.decoder.layers.17",
        #     # "model.decoder.layers.18",
        #     # "model.decoder.layers.19",
        #     # "model.decoder.layers.20",
        #     # "model.decoder.layers.21",
        #     # "model.decoder.layers.22",
        #     # "model.decoder.layers.23",
        # ],
        "control_scheme": [
            "model.decoder.layers.0",
            "model.decoder.layers.1",
            "model.decoder.layers.2",
            "model.decoder.layers.3",
            "model.decoder.layers.4",
            "model.decoder.layers.5",
            # "model.decoder.layers.6",
            # "model.decoder.layers.7",
            # "model.decoder.layers.8",
            # "model.decoder.layers.9",
            # "model.decoder.layers.10",
            # "model.decoder.layers.11",
            # "model.decoder.layers.12",
            # "model.decoder.layers.13",
            # "model.decoder.layers.14",
            # "model.decoder.layers.15",
            # "model.decoder.layers.16",
            # "model.decoder.layers.17",
            # "model.decoder.layers.18",
            # "model.decoder.layers.19",
            # "model.decoder.layers.20",
            # "model.decoder.layers.21",
            # "model.decoder.layers.22",
            # "model.decoder.layers.23",
        ],
        "transformer_scheme": [
            "model.decoder.layers.0",
            "model.decoder.layers.1",
            "model.decoder.layers.2",
            "model.decoder.layers.3",
            "model.decoder.layers.4",
            "model.decoder.layers.5",
            "model.decoder.layers.6",
            "model.decoder.layers.7",
            "model.decoder.layers.8",
            "model.decoder.layers.9",
            "model.decoder.layers.10",
            "model.decoder.layers.11",
            "model.decoder.layers.12",
            "model.decoder.layers.13",
            "model.decoder.layers.14",
            "model.decoder.layers.15",
            "model.decoder.layers.16",
            "model.decoder.layers.17",
            "model.decoder.layers.18",
            "model.decoder.layers.19",
            "model.decoder.layers.20",
            "model.decoder.layers.21",
            "model.decoder.layers.22",
            "model.decoder.layers.23",
        ],
        "embedding_scheme": [
            "model.decoder.embed_tokens",
            "model.decoder.embed_positions",
        ],
    },
}


class classifier_distilbert(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(self, hidden_state):
        pooled_output = hidden_state[:, 0]
        pooled_output = self.model.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=(),
            attentions=(),
        )


class classifier_bert(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(self, sequence_output):
        pooled_output = (
            self.model.bert.pooler(sequence_output)
            if self.model.bert.pooler is not None
            else None
        )
        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=(),
            hidden_states=(),
            attentions=(),
            cross_attentions=(),
        )
        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class classifier_roberta(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(self, sequence_output):
        pooled_output = (
            self.model.roberta.pooler(sequence_output)
            if self.model.roberta.pooler is not None
            else None
        )
        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=(),
            hidden_states=(),
            attentions=(),
            cross_attentions=(),
        )
        pooled_output = outputs[0]
        logits = self.model.classifier(pooled_output)
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class classifier_opt(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(self, hidden_states, input_ids):
        if self.model.model.decoder.final_layer_norm is not None:
            hidden_states = self.model.model.decoder.final_layer_norm(hidden_states)
        logits = self.model.score(hidden_states)

        batch_size, sequence_length = input_ids.shape[:2]
        if self.model.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(input_ids, self.model.config.pad_token_id).int().argmax(-1)
                    - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]
        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=pooled_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
