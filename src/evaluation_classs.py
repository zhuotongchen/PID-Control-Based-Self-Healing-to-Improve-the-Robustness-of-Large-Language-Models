import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer


from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import logging
from transformers.trainer_pt_utils import nested_detach


logger = logging.get_logger(__name__)


class evaluator(Trainer):
    @torch.no_grad()
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
        predictions = logits.argmax(dim=1)
        correctness = (predictions == labels).sum()
        accuracy = correctness / len(labels)
        return (accuracy, model_output) if return_outputs else accuracy

    # def prediction_step(
    #     self,
    #     model: nn.Module,
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     has_labels = (
    #         False
    #         if len(self.label_names) == 0
    #         else all(inputs.get(k) is not None for k in self.label_names)
    #     )
    #     # For CLIP-like models capable of returning loss values.
    #     # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
    #     # is `True` in `model.forward`.
    #     return_loss = inputs.get("return_loss", None)
    #     if return_loss is None:
    #         return_loss = self.can_return_loss
    #     loss_without_labels = (
    #         True if len(self.label_names) == 0 and return_loss else False
    #     )

    #     inputs = self._prepare_inputs(inputs)
    #     if ignore_keys is None:
    #         if hasattr(self.model, "config"):
    #             ignore_keys = getattr(
    #                 self.model.config, "keys_to_ignore_at_inference", []
    #             )
    #         else:
    #             ignore_keys = []

    #     # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
    #     if has_labels or loss_without_labels:
    #         labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
    #         if len(labels) == 1:
    #             labels = labels[0]
    #     else:
    #         labels = None

    #     if has_labels or loss_without_labels:
    #         with self.compute_loss_context_manager():
    #             loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
    #         loss = loss.mean().detach()

    #         if isinstance(outputs, dict):
    #             logits = tuple(
    #                 v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
    #             )
    #         else:
    #             logits = outputs[1:]
    #     else:
    #         loss = None
    #         with self.compute_loss_context_manager():
    #             outputs = model(**inputs)
    #         if isinstance(outputs, dict):
    #             logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
    #         else:
    #             logits = outputs

    #     if prediction_loss_only:
    #         return (loss, None, None)

    #     logits = nested_detach(logits)
    #     if len(logits) == 1:
    #         logits = logits[0]

    #     return (loss, logits, labels)
