import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .utils import TRANSFORMERS_MODELS_TO_EMBEDDING_MODULES_MAPPING
from .utils import (
    classifier_distilbert,
    classifier_bert,
    classifier_roberta,
    classifier_opt,
)

from transformers.utils import is_accelerate_available

if is_accelerate_available():
    from accelerate import Accelerator

from peft.peft_model import PeftModel

import sys


class closed_loop_control_module(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        ccl_config: dict,
        model_name: str,
    ) -> None:
        super().__init__()

        model_config = getattr(model, "config", {"model_type": "custom"})
        model_config = model_config.to_dict()
        if (
            model_config["model_type"] == "roberta"
            and model_name.split("-")[-1] == "base"
        ):
            model_config["model_type"] = "roberta-base"
        ccl_config = self._prepare_config(ccl_config, model_config)

        self.model_type = model_config["model_type"]

        if isinstance(model, PeftModel):
            model = model.base_model.model

        key_list = [key for key, _ in model.named_modules()]

        if any(
            key not in key_list
            for key in ccl_config["target_modules"]["transformer_scheme"]
        ):
            raise ValueError(
                f"Target modules not found in the base model. "
                f"Please check the target modules and try again."
            )

        if any(
            key not in ccl_config["target_modules"]["transformer_scheme"]
            for key in ccl_config["target_modules"]["control_scheme"]
        ):
            raise ValueError(
                f"Target modules not found in the base model. "
                f"Please check the target modules and try again."
            )

        self.P_control = False
        self.I_control = False
        self.D_control = False
        self.control_scheme = ccl_config["control_scheme"].split("-")
        self.temporal_control = False
        self.embedding_control = True

        self.temporal_orthgonal_projection = dict()
        self.embedding_orthgonal_projection = dict()

        self.temporal_projection_subspace = dict()
        self.embedding_projection_subspace = dict()

        (
            self.embedding,
            self.sequential_model,
            self.classification,
        ) = self.construct_sequential_model(model, ccl_config)

        if self.model_type == "bert":
            self.get_extended_attention_mask = model.bert.get_extended_attention_mask
        elif self.model_type == "roberta" or self.model_type == "roberta-base":
            self.get_extended_attention_mask = model.roberta.get_extended_attention_mask
        else:
            self.get_extended_attention_mask = None

        if self.model_type == "opt":
            self.final_layer_norm = model.model.decoder.final_layer_norm
        self.ccl_config = ccl_config

        self.config = model.config
        self.get_input_embeddings = model.get_input_embeddings

    def device(self):
        return self.embedding.word_embeddings.weight.device

    def projection(
        self, x, attention_mask, temporal_projection_matrix, embedding_projection_matrix
    ):
        if attention_mask.dim() < x.dim():
            x = x * attention_mask.unsqueeze(-1)

        if self.temporal_control is True:
            temporal_control = (
                -torch.tensordot(
                    x,
                    temporal_projection_matrix,
                    dims=[[1], [0]],
                )
                .transpose(1, 2)
                .contiguous()
            )
        else:
            temporal_control = torch.zeros_like(x)

        if self.embedding_control is True:
            embedding_control = -torch.tensordot(
                x,
                embedding_projection_matrix,
                dims=[[2], [0]],
            )
        else:
            embedding_control = torch.zeros_like(x)
            
        control = temporal_control + embedding_control
        control = control.norm(p=2, dim=[1, 2]).mean()
        print("Control mag:", control)
        return temporal_control + embedding_control

    def evaluate_loss(
        self, x, attention_mask, temporal_projection_matrix, embedding_projection_matrix
    ):
        if attention_mask.dim() < x.dim():
            x = x * attention_mask.unsqueeze(-1)

        if self.temporal_control is True:
            x_temporal_projection = (
                torch.tensordot(
                    x,
                    temporal_projection_matrix,
                    dims=[[1], [0]],
                )
                .transpose(1, 2)
                .contiguous()
            )
            loss_temporal = nn.MSELoss()(x, x_temporal_projection)
        else:
            loss_temporal = nn.MSELoss()(x, x)

        if self.embedding_control is True:
            x_embedding_projection = torch.tensordot(
                x,
                embedding_projection_matrix,
                dims=[[2], [0]],
            )
            loss_embedding = nn.MSELoss()(x, x_embedding_projection)
        else:
            loss_embedding = nn.MSELoss()(x, x)
        return loss_temporal + loss_embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        if (
            self.embedding is None
            or self.sequential_model is None
            or self.classification is None
        ):
            raise ValueError(
                "Please construct all embedding, sequential_model, classification"
            )

        if any(
            name not in self.temporal_orthgonal_projection
            for name in self.ccl_config["target_modules"]["control_scheme"]
        ) or any(
            name not in self.embedding_orthgonal_projection
            for name in self.ccl_config["target_modules"]["control_scheme"]
        ):
            raise ValueError("Please construct embedding_matrix")

        # controlled forward propagation
        # generate embeddings
        if self.model_type == "opt":
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            word_embeddings = self.embedding[0](input_ids)
        else:
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
        elif self.model_type == "opt":
            batch_size, seq_length = input_shape
            past_key_values_length = 0
            # embed positions
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, word_embeddings, past_key_values_length
            )
            pos_embeds = self.embedding[1](attention_mask, past_key_values_length)
            hidden_state = word_embeddings + pos_embeds

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
            elif self.model_type == "opt":
                hidden_state = submodel(
                    hidden_states=hidden_state,
                    attention_mask=causal_attention_mask,
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
        if self.model_type == "opt":
            model_output = self.classification(hidden_state, input_ids)
        else:
            model_output = self.classification(hidden_state)
        sys.exit()
        return model_output

    @staticmethod
    def _prepare_config(ccl_config: dict, model_config: dict):
        if "model_type" not in ccl_config:
            ccl_config["model_type"] = model_config["model_type"]
        if ccl_config["target_modules"] is None:
            ccl_config[
                "target_modules"
            ] = TRANSFORMERS_MODELS_TO_EMBEDDING_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return ccl_config

    @staticmethod
    def construct_sequential_model(model, config):
        # construct nn.sequential model based on target_modules
        if config["model_type"] == "opt":
            embedding = [
                model.get_submodule(module_name)
                for module_name in config["target_modules"]["embedding_scheme"]
            ]
        else:
            embedding = model.get_submodule(
                config["target_modules"]["embedding_scheme"]
            )
        sequential_model = nn.Sequential(
            *[
                model.get_submodule(key)
                for key in config["target_modules"]["transformer_scheme"]
            ]
        )
        if config["model_type"] == "distilbert":
            classifier = classifier_distilbert(model)
        elif config["model_type"] == "bert":
            classifier = classifier_bert(model)
        elif (
            config["model_type"] == "roberta" or config["model_type"] == "roberta-base"
        ):
            classifier = classifier_roberta(model)
        elif config["model_type"] == "opt":
            classifier = classifier_opt(model)
        return embedding, sequential_model, classifier

    @staticmethod
    @torch.no_grad()
    def svd(M):
        variance = 0.99

        # M = M[:1000]
        M = M.transpose(0, 2).contiguous().view(M.shape[2], -1)

        U, S, V = torch.svd(M)
        # Calculate the normalized cumulative sum of the squared singular values
        cumulative_variance = torch.cumsum(S**2, dim=0)
        total_variance = cumulative_variance[-1]
        normalized_cumulative_variance = cumulative_variance / total_variance

        # Determine the number of singular values to keep for the top p percent
        num_singular_values = torch.sum(
            normalized_cumulative_variance <= variance
        ).item()
        if num_singular_values == 0:
            num_singular_values = 1

        U_top = U[:, :num_singular_values]
        S_top = S[:num_singular_values]
        V_top = V[:, :num_singular_values]

        error = nn.MSELoss()(U_top.mm(torch.diag(S_top)).mm(V_top.t()), M)
        return U_top, round(error.item(), 3)

    @staticmethod
    @torch.no_grad()
    def hosvd(M):
        variance = 0.99

        mode_m_basis = []
        for mode in range(1, 3):
            U, S, V = torch.svd(
                M.transpose(0, mode).contiguous().view(M.shape[mode], -1)
            )

            # Calculate the normalized cumulative sum of the squared singular values
            cumulative_variance = torch.cumsum(S**2, dim=0)
            total_variance = cumulative_variance[-1]
            normalized_cumulative_variance = cumulative_variance / total_variance

            # Determine the number of singular values to keep for the top p percent
            num_singular_values = torch.sum(
                normalized_cumulative_variance <= variance
            ).item()
            if num_singular_values == 0:
                num_singular_values = 1

            mode_m_basis.append(U[:, :num_singular_values])

        # Tucker decomposition reconstruction
        core = torch.tensordot(
            torch.tensordot(M, mode_m_basis[0], dims=([1], [0])),
            mode_m_basis[1],
            dims=([1], [0]),
        )
        M_reconstruct = torch.tensordot(
            torch.tensordot(core, mode_m_basis[0].t(), dims=([1], [0])),
            mode_m_basis[1].t(),
            dims=([1], [0]),
        )
        error = nn.MSELoss()(M, M_reconstruct)
        return mode_m_basis, round(error.item(), 3)

    @torch.no_grad()
    def construct_embedding_module(
        self,
        embedding,
        sequential_model,
        classification_model,
        config,
        dataset,
        data_collator,
    ):
        # construct data loader
        batch_size = config["batch_size"]
        accelerator = Accelerator()
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "pin_memory": True,
        }
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = RandomSampler(dataset)
        data_loader = accelerator.prepare(DataLoader(dataset, **dataloader_params))

        # construct state trajectory
        state_trajectory = dict()
        if config["target_modules"]["control_word_embedding"]:
            state_trajectory["word_embedding"] = []
        for name in config["target_modules"]["control_scheme"]:
            state_trajectory[name] = []

        num_accumulated_samples, num_searched_samples = 0, 0
        for step, inputs in enumerate(data_loader):
            input_ids, attention_mask, labels = (
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["labels"],
            )

            if self.model_type == "opt":
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                word_embeddings = embedding[0](input_ids)
            else:
                word_embeddings = embedding.word_embeddings(input_ids)
            if "word_embedding" in state_trajectory:
                state_trajectory["word_embedding"].append(
                    word_embeddings.cpu()
                    if word_embeddings.device.type == "cuda"
                    else word_embeddings
                )

            if self.model_type == "distilbert":
                hidden_state = embedding(input_ids=None, input_embeds=word_embeddings)

            elif (
                self.model_type == "bert"
                or self.model_type == "roberta-base"
                or self.model_type == "roberta"
            ):
                input_shape = word_embeddings.size()[:-1]
                batch_size, seq_length = input_shape
                buffered_token_type_ids = embedding.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
                hidden_state = embedding(
                    input_ids=None,
                    position_ids=None,
                    token_type_ids=token_type_ids,
                    inputs_embeds=word_embeddings,
                    past_key_values_length=0,
                )
                extended_attention_mask: torch.Tensor = (
                    self.get_extended_attention_mask(attention_mask, input_shape)
                )
            elif self.model_type == "opt":
                batch_size, seq_length = input_shape
                past_key_values_length = 0
                # embed positions
                causal_attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, input_shape, word_embeddings, past_key_values_length
                )
                pos_embeds = embedding[1](attention_mask, past_key_values_length)
                hidden_state = word_embeddings + pos_embeds

            for submodel_name, submodel in zip(
                config["target_modules"]["transformer_scheme"], sequential_model
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
                elif self.model_type == "opt":
                    hidden_state = submodel(
                        hidden_states=hidden_state,
                        attention_mask=causal_attention_mask,
                    )

                if isinstance(hidden_state, tuple):
                    hidden_state = hidden_state[0]

                if submodel_name in state_trajectory:
                    hidden_state_masked = hidden_state * attention_mask.unsqueeze(-1)
                    state_trajectory[submodel_name].append(
                        hidden_state_masked.cpu()
                        if hidden_state_masked.device.type == "cuda"
                        else hidden_state_masked
                    )

            # save only correctly predicted trajectory
            if self.model_type == "opt":
                model_outputs = classification_model(hidden_state, input_ids)
            else:
                model_outputs = classification_model(hidden_state)
            logits = model_outputs.logits
            correctness = logits.argmax(dim=1) == labels
            correctness = (
                correctness.cpu() if correctness.device.type == "cuda" else correctness
            )
            for submodel_name in state_trajectory:
                state_trajectory[submodel_name][-1] = state_trajectory[submodel_name][
                    -1
                ][correctness]
            num_accumulated_samples += correctness.sum()
            num_searched_samples += len(correctness)
            if num_accumulated_samples > 1000:
                break

        print(
            "Construct controllers: Number of samples:",
            round(num_accumulated_samples.item(), 0),
            "Number of searched samples:",
            num_searched_samples,
        )

        device = logits.device
        for control_ in self.control_scheme:
            self.contruct_controller_projection(
                state_trajectory=state_trajectory,
                control_type=control_,
                device=device,
            )

        for name in self.temporal_orthgonal_projection:
            self.temporal_orthgonal_projection[name] /= len(self.control_scheme)
            self.embedding_orthgonal_projection[name] /= len(self.control_scheme)

    def contruct_controller_projection(
        self, state_trajectory, control_type, device=None
    ):
        # construct optimal control regularizations
        control_regularization = self.ccl_config["control_regularization"]
        optimal_regularizations = []
        lambda_t = 0.0
        for _ in range(
            len(self.ccl_config["target_modules"]["control_scheme"]), -1, -1
        ):
            optimal_regularizations.append(
                control_regularization / (1.0 + lambda_t + control_regularization)
            )
            lambda_t = (
                control_regularization
                * (1.0 + lambda_t)
                / (1.0 + control_regularization + lambda_t)
            )
        optimal_regularizations = optimal_regularizations[::-1]

        for ii, (name, state) in enumerate(state_trajectory.items()):
            concat_states = (
                torch.cat(state).to(device) if device is not None else torch.cat(state)
            )
            # construct projection matrix for P controller
            if control_type == "P":
                M = concat_states
            # construct projection matrix for I controller
            elif control_type == "I":
                if ii == 0:
                    M = concat_states
                else:
                    M = (M * ii + concat_states) / (ii + 1)
            # construct projection matrix for D controller
            elif control_type == "D":
                submodule_names = list(state_trajectory.keys())
                if ii == 0:
                    M = concat_states
                else:
                    M_past = (
                        torch.cat(state_trajectory[submodule_names[ii - 1]]).to(device)
                        if device is not None
                        else torch.cat(state_trajectory[submodule_names[ii - 1]])
                    )
                    M = concat_states - M_past
            else:
                raise ValueError(f"Unknown control type. ")

            basis, error = self.hosvd(M)
            print(
                "Controller:",
                control_type,
                "Layer:",
                name,
                "Temp Rank:",
                basis[0].shape[1],
                "/",
                basis[0].shape[0],
                "Embedding Rank:",
                basis[1].shape[1],
                "/",
                basis[1].shape[0],
                "Error:",
                error,
            )

            device = M.device

            temporal_diagonal_factor = torch.ones(basis[0].shape[0])
            temporal_diagonal_factor[-basis[0].shape[1] :] -= optimal_regularizations[
                ii
            ]
            temporal_diagonal_factor = torch.diag(temporal_diagonal_factor).to(device)
            temporal_orthgonal_projection = temporal_diagonal_factor - basis[0].mm(
                basis[0].t()
            )
            if name not in self.temporal_orthgonal_projection:
                self.temporal_orthgonal_projection[name] = temporal_orthgonal_projection
            else:
                self.temporal_orthgonal_projection[name] = (
                    self.temporal_orthgonal_projection[name]
                    + temporal_orthgonal_projection
                )

            embedding_diagonal_factor = torch.ones(basis[1].shape[0])
            embedding_diagonal_factor[-basis[1].shape[1] :] -= optimal_regularizations[
                ii
            ]
            embedding_diagonal_factor = torch.diag(embedding_diagonal_factor).to(device)
            embedding_orthgonal_projection = embedding_diagonal_factor - basis[1].mm(
                basis[1].t()
            )
            if name not in self.embedding_orthgonal_projection:
                self.embedding_orthgonal_projection[
                    name
                ] = embedding_orthgonal_projection
            else:
                self.embedding_orthgonal_projection[name] = (
                    self.embedding_orthgonal_projection[name]
                    + embedding_orthgonal_projection
                )

            # orthogonal projection onto embedding subspace
            if name not in self.temporal_projection_subspace:
                self.temporal_projection_subspace[name] = basis[0].mm(basis[0].t())
            else:
                self.temporal_projection_subspace[
                    name
                ] = self.temporal_projection_subspace[name] + basis[0].mm(basis[0].t())

            if name not in self.embedding_projection_subspace:
                self.embedding_projection_subspace[name] = basis[1].mm(basis[1].t())
            else:
                self.embedding_projection_subspace[
                    name
                ] = self.embedding_projection_subspace[name] + basis[1].mm(basis[1].t())
