import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .utils import TRANSFORMERS_MODELS_TO_EMBEDDING_MODULES_MAPPING
from .utils import classifier_distilbert, classifier_bert, classifier_roberta

from transformers.utils import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator

from peft.peft_model import PeftModel

from src.ccl.closed_loop_control_module import closed_loop_control_module


class pmp_control_module(closed_loop_control_module):
    def __init__(
        self,
        model: PreTrainedModel,
        ccl_config: dict,
        model_name: str,
    ) -> None:
        super().__init__(model=model, ccl_config=ccl_config, model_name=model_name)

        # PMP parameters
        self._pmp_max_ite = 3
        self._lr = 0.1
        # self._epsilon = opt.epsilon
        # self._lr_factor = opt.lr_factor
        # self._lr = self._epsilon / self._lr_factor
        # self._pmp_radius = torch.tensor(self._epsilon).cuda()
        # self._pmp_norm = opt.norm

    @torch.no_grad()
    def initialize_control(self, input_ids, attention_mask, **kwargs):
        controls_all = dict()
        # generate embeddings
        word_embeddings = self.embedding.word_embeddings(input_ids)
        if (
            "word_embedding" in self.temporal_orthgonal_projection
            and "word_embedding" in self.embedding_orthgonal_projection
        ):
            u_word_embeddings = self.projection(
                word_embeddings,
                attention_mask,
                self.temporal_orthgonal_projection["word_embedding"],
                self.embedding_orthgonal_projection["word_embedding"],
            )
            word_embeddings = word_embeddings + u_word_embeddings
            controls_all["word_embedding"] = u_word_embeddings

        if self.model_type == "distilbert":
            hidden_state = self.embedding(input_ids=None, input_embeds=word_embeddings)

        elif (
            self.model_type == "bert"
            or self.model_type == "roberta-base"
            or self.model_type == "roberta"
        ):
            input_shape = word_embeddings.size()[:-1]
            batch_size, seq_length = input_shape
            buffered_token_type_ids = self.embedding.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                batch_size, seq_length
            )
            token_type_ids = buffered_token_type_ids_expanded
            hidden_state = self.embedding(
                input_ids=None,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=word_embeddings,
                past_key_values_length=0,
            )
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape
            )

        for layer_name, submodel in zip(
            self.ccl_config["target_modules"]["transformer_scheme"],
            self.sequential_model,
        ):
            if self.model_type == "distilbert":
                hidden_state = submodel(x=hidden_state, attn_mask=attention_mask)
            elif (
                self.model_type == "bert"
                or self.model_type == "roberta-base"
                or self.model_type == "roberta"
            ):
                hidden_state = submodel(
                    hidden_states=hidden_state,
                    attention_mask=extended_attention_mask,
                )

            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]

            if layer_name in self.ccl_config["target_modules"]["control_scheme"]:
                u = self.projection(
                    hidden_state,
                    attention_mask,
                    self.temporal_orthgonal_projection[layer_name],
                    self.embedding_orthgonal_projection[layer_name],
                )
                hidden_state = hidden_state + u
                controls_all[layer_name] = u
        return controls_all

    def forward(self, input_ids, attention_mask, **kwargs):
        # initialize controls
        controls_all = self.initialize_control(input_ids, attention_mask, **kwargs)
        for layer_name, control in controls_all.items():
            control.detach_()
            control.requires_grad_(True)
        # construct optimizer
        optimizer = torch.optim.SGD(
            [controls_all[name] for name in controls_all],
            lr=self._lr,
            momentum=0.9,
            weight_decay=1.,
        )
        # pmp loop
        with torch.enable_grad():
            for iteration in range(self._pmp_max_ite):
                optimizer.zero_grad()
                # embedding loss to be optimized
                loss = 0.0

                word_embeddings = self.embedding.word_embeddings(input_ids)
                if (
                    "word_embedding" in self.temporal_projection_subspace
                    and "word_embedding" in self.embedding_projection_subspace
                ):
                    word_embeddings = word_embeddings + controls_all["word_embedding"]
                    loss = loss + self.evaluate_loss(
                        word_embeddings,
                        attention_mask,
                        self.temporal_projection_subspace["word_embedding"],
                        self.embedding_projection_subspace["word_embedding"],
                    )

                if self.model_type == "distilbert":
                    hidden_state = self.embedding(
                        input_ids=None, input_embeds=word_embeddings
                    )

                elif (
                    self.model_type == "bert"
                    or self.model_type == "roberta-base"
                    or self.model_type == "roberta"
                ):
                    input_shape = word_embeddings.size()[:-1]
                    batch_size, seq_length = input_shape
                    buffered_token_type_ids = self.embedding.token_type_ids[
                        :, :seq_length
                    ]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                        batch_size, seq_length
                    )
                    token_type_ids = buffered_token_type_ids_expanded
                    hidden_state = self.embedding(
                        input_ids=None,
                        position_ids=None,
                        token_type_ids=token_type_ids,
                        inputs_embeds=word_embeddings,
                        past_key_values_length=0,
                    )
                    extended_attention_mask: torch.Tensor = (
                        self.get_extended_attention_mask(attention_mask, input_shape)
                    )

                for layer_name, submodel in zip(
                    self.ccl_config["target_modules"]["transformer_scheme"],
                    self.sequential_model,
                ):
                    if self.model_type == "distilbert":
                        hidden_state = submodel(
                            x=hidden_state, attn_mask=attention_mask
                        )
                    elif (
                        self.model_type == "bert"
                        or self.model_type == "roberta-base"
                        or self.model_type == "roberta"
                    ):
                        hidden_state = submodel(
                            hidden_states=hidden_state,
                            attention_mask=extended_attention_mask,
                        )

                    if isinstance(hidden_state, tuple):
                        hidden_state = hidden_state[0]

                    if (
                        layer_name
                        in self.ccl_config["target_modules"]["control_scheme"]
                    ):
                        hidden_state = hidden_state + controls_all[layer_name]

                        loss = loss + self.evaluate_loss(
                            hidden_state,
                            attention_mask,
                            self.temporal_projection_subspace[layer_name],
                            self.embedding_projection_subspace[layer_name],
                        )

                loss.backward()
                optimizer.step()
        with torch.no_grad():
            model_output = self.classification(hidden_state)
        return model_output
