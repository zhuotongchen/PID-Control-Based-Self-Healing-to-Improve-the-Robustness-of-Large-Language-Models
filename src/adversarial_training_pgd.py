import torch
import torch.nn as nn

from transformers import Trainer
from transformers.models.distilbert.modeling_distilbert import (
    BaseModelOutput,
    SequenceClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.trainer_pt_utils import nested_detach
from typing import Any, Dict, List, Optional, Tuple, Union


class Trainer_adversarial_training(Trainer):
    def __init__(
        self,
        train_method="pgd",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_search_steps = 2
        self.search_step_size = 1e-1
        self.freelb_inner_steps = 3
        self.train_method = train_method
        self.rand_init_mag = 1e-2
        self.grad_square = False
        self.max_norm = 2e-1
        self.norm_method = "l2"

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        if has_labels or loss_without_labels:
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            if isinstance(outputs, dict):
                logits = tuple(
                    v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
                )
            else:
                logits = outputs[1:]
        else:
            loss = None
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, attention_mask, labels = (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["labels"],
        )
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = (
            model_output["logits"]
            if isinstance(model_output, dict)
            else model_output[0]
        )
        loss_standard = nn.CrossEntropyLoss()(logits, labels)

        if self.train_method == "pgd":
            loss_adversarial, _ = self.std_adv_train_step(model, inputs, return_outputs)
        elif self.train_method == "freelb":
            loss_adversarial, _ = self.freelb_train_step(model, inputs, return_outputs)

        alpha = 0.5
        loss = alpha * loss_standard + (1.0 - alpha) * loss_adversarial
        return (loss, model_output) if return_outputs else loss

    def std_adv_train_step(self, model, inputs, return_outputs=False):
        model.train()
        input_ids, attention_mask, labels = (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["labels"],
        )
        if model.config.model_type == "distilbert":
            embedding_function = (
                model.base_model.model.distilbert.embeddings.word_embeddings
            )
        elif model.config.model_type == "bert":
            embedding_function = model.base_model.model.bert.embeddings.word_embeddings
        elif model.config.model_type == "roberta":
            embedding_function = (
                model.base_model.model.roberta.embeddings.word_embeddings
            )
        else:
            raise ValueError(f"Unknown model. ")

        embeds_init = embedding_function(input_ids)
        dim_temporal, dim_embedding = input_ids.size()[-1], embeds_init.size()[-1]
        delta = torch.zeros_like(embeds_init).uniform_(
            -1, 1
        ) * attention_mask.unsqueeze(2)
        dims = torch.FloatTensor([dim_temporal * dim_embedding]).to(delta.device)
        mag = self.rand_init_mag / torch.sqrt(dims)
        delta = (delta * mag.view(-1, 1, 1)).detach()
        delta.requires_grad_()

        model_outputs = model(
            inputs_embeds=delta + embeds_init, attention_mask=attention_mask
        )
        logits = model_outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)

        for t_adv in range(self.num_search_steps):
            delta_grad = torch.autograd.grad(loss, delta)[0].detach()
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(
                -1, 1, 1
            )

            if self.grad_square:
                denorm = denorm**2
            delta = (delta + self.search_step_size * delta_grad / denorm).detach()

            if self.max_norm > 0:
                delta_norm = (
                    torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1)
                    .to(embeds_init)
                    .detach()
                )
                exceed_mask = (delta_norm > self.max_norm).to(embeds_init)
                delta = (
                    delta
                    * (self.max_norm / delta_norm * exceed_mask + (1 - exceed_mask))
                    .view(-1, 1, 1)
                    .detach()
                )

            delta.requires_grad_()
            adv_embeds_init = embedding_function(input_ids)
            model_outputs = model(
                inputs_embeds=delta + adv_embeds_init, attention_mask=attention_mask
            )
            logits = model_outputs.logits
            loss = nn.CrossEntropyLoss()(logits, labels)

        return loss, model_outputs

    def freelb_train_step(self, model, inputs, return_outputs=False):
        model.train()
        input_ids, attention_mask, labels = (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["labels"],
        )
        if model.config.model_type == "distilbert":
            embedding_function = (
                model.base_model.model.distilbert.embeddings.word_embeddings
            )
        elif model.config.model_type == "bert":
            embedding_function = model.base_model.model.bert.embeddings.word_embeddings
        elif model.config.model_type == "roberta":
            embedding_function = (
                model.base_model.model.roberta.embeddings.word_embeddings
            )
        else:
            raise ValueError(f"Unknown model. ")

        embeds_init = embedding_function(input_ids)
        dim_temporal, dim_embedding = input_ids.size()[-1], embeds_init.size()[-1]
        delta = torch.zeros_like(embeds_init).uniform_(
            -1, 1
        ) * attention_mask.unsqueeze(2)
        dims = torch.FloatTensor([dim_temporal * dim_embedding]).to(delta.device)
        mag = self.rand_init_mag / torch.sqrt(dims)
        delta = (delta * mag.view(-1, 1, 1)).detach()
        delta.requires_grad_()

        model_outputs, first_layer_out = self.forward_freelb(
            model, delta + embeds_init, attention_mask
        )
        logits = model_outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss = loss / (1 + self.num_search_steps)

        for t_adv in range(self.num_search_steps):
            p_var = torch.autograd.grad(loss, first_layer_out)[0].clone().detach()
            const_embeds_init = embedding_function(input_ids).detach()
            delta = self.freelb_inner_step(
                p_var, delta, const_embeds_init, attention_mask, model
            )
            adv_embeds_init = embedding_function(input_ids).detach()

            model_outputs, first_layer_out = self.forward_freelb(
                model, delta + adv_embeds_init, attention_mask
            )
            logits = model_outputs.logits
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss = loss / (1 + self.num_search_steps)

        return loss, model_outputs

    def freelb_inner_step(self, p_var, delta, embeds_init, attention_mask, model):
        for t in range(self.freelb_inner_steps):
            hal = self.Hamiltonian_fwd(
                model=model,
                p_var=p_var,
                inputs_embeds=embeds_init + delta,
                attention_mask=attention_mask,
            )
            delta_grad = torch.autograd.grad(
                hal, delta, only_inputs=True, retain_graph=False
            )[0]

            denorm = torch.clamp(
                torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(
                    -1, 1, 1
                ),
                min=1e-10,
            )

            delta = (delta + self.search_step_size * delta_grad / denorm).detach()
            if self.max_norm > 0:
                delta_norm = torch.norm(
                    delta.view(delta.size(0), -1), p=2, dim=1
                ).detach()
                exceed_mask = (delta_norm > self.max_norm).to(embeds_init)
                delta = (
                    delta
                    * (self.max_norm / delta_norm * exceed_mask + (1 - exceed_mask))
                    .view(-1, 1, 1)
                    .detach()
                )
            delta.requires_grad_()
        return delta

    def Hamiltonian_fwd(self, model, p_var, inputs_embeds, attention_mask):
        if model.config.model_type == "distilbert":
            hidden_state = model.base_model.model.distilbert.embeddings(
                input_ids=None, input_embeds=inputs_embeds
            )
            hidden_state = model.base_model.model.distilbert.transformer.layer[0](
                x=hidden_state,
                attn_mask=attention_mask,
                head_mask=None,
                output_attentions=False,
            )[-1]
        elif model.config.model_type == "bert" or model.config.model_type == "roberta":
            encoder = (
                model.base_model.model.bert
                if model.config.model_type == "bert"
                else model.base_model.model.roberta
            )

            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            buffered_token_type_ids = encoder.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                batch_size, seq_length
            )
            token_type_ids = buffered_token_type_ids_expanded
            hidden_state = encoder.embeddings(
                input_ids=None,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=0,
            )
            extended_attention_mask: torch.Tensor = encoder.get_extended_attention_mask(
                attention_mask, input_shape
            )
            hidden_state = encoder.encoder.layer[0](
                hidden_states=hidden_state, attention_mask=extended_attention_mask
            )[0]
        return torch.sum(hidden_state * p_var)

    def forward_freelb(self, model, inputs_embeds, attention_mask):
        if model.config.model_type == "distilbert":
            return self.forward_distilbert(model, inputs_embeds, attention_mask)
        elif model.config.model_type == "bert" or model.config.model_type == "roberta":
            return self.forward_bert(model, inputs_embeds, attention_mask)
        else:
            raise ValueError(f"Unknown model. ")

    def forward_distilbert(self, model, inputs_embeds, attention_mask):
        x = model.base_model.model.distilbert.embeddings(
            input_ids=None, input_embeds=inputs_embeds
        )

        hidden_state = x
        for i, layer_module in enumerate(
            model.base_model.model.distilbert.transformer.layer
        ):
            layer_outputs = layer_module(
                x=hidden_state,
                attn_mask=attention_mask,
                head_mask=None,
                output_attentions=False,
            )

            hidden_state = layer_outputs[-1]

            if i == 0:
                first_layer_out = hidden_state
                first_layer_out.requires_grad_()
                first_layer_out.retain_grad()
                hidden_state = first_layer_out

        distilbert_output = BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=(), attentions=()
        )

        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = model.base_model.model.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = model.base_model.model.dropout(pooled_output)
        logits = model.base_model.model.classifier(pooled_output)
        return (
            SequenceClassifierOutput(
                loss=None,
                logits=logits,
                hidden_states=distilbert_output.hidden_states,
                attentions=distilbert_output.attentions,
            ),
            first_layer_out,
        )

    def forward_bert(self, model, inputs_embeds, attention_mask):
        if model.config.model_type == "bert":
            encoder = model.base_model.model.bert
        elif model.config.model_type == "roberta":
            encoder = model.base_model.model.roberta

        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        buffered_token_type_ids = encoder.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
            batch_size, seq_length
        )
        token_type_ids = buffered_token_type_ids_expanded
        x = encoder.embeddings(
            input_ids=None,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )
        extended_attention_mask: torch.Tensor = encoder.get_extended_attention_mask(
            attention_mask, input_shape
        )
        hidden_states = x
        for i, layer_module in enumerate(encoder.encoder.layer):
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
            )

            hidden_states = layer_outputs[0]

            if i == 0:
                first_layer_out = hidden_states
                first_layer_out.requires_grad_()
                first_layer_out.retain_grad()
                hidden_states = first_layer_out

        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            encoder.pooler(sequence_output) if encoder.pooler is not None else None
        )
        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

        if model.config.model_type == "bert":
            pooled_output = outputs[1]
            pooled_output = model.base_model.model.dropout(pooled_output)
        elif model.config.model_type == "roberta":
            pooled_output = outputs[0]
        logits = model.base_model.model.classifier(pooled_output)
        return (
            SequenceClassifierOutput(
                loss=None,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ),
            first_layer_out,
        )
