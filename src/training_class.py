import torch.nn as nn
from transformers import Trainer


class Trainer_standard(Trainer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, model_output) if return_outputs else loss
