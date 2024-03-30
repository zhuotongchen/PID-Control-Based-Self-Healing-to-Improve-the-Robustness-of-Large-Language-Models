#!/usr/bin/env python3
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import sys

import numpy as np
import argparse
import random
import time

import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.append("src")

from src.data_module import DataModule
from src.ccl.closed_loop_control_module import closed_loop_control_module
from src.ccl.pmp_control_module import pmp_control_module

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from src.evaluation_classs import evaluator
from collections import OrderedDict


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# model parameters
parser.add_argument(
    "--model_name",
    type=str,
    default="facebook/opt-1.3b",
    help="model configuration",
    choices=[
        "distilbert-base-uncased",
        "bert-large-uncased",
        "roberta-large",
        "roberta-base",
        "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "facebook/opt-1.3b",
    ],
)
parser.add_argument(
    "--pretrained_checkpoint", type=str, default=None, help="models are saved here"
)
parser.add_argument("--lora", default=True, action="store_false")
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
# data parameters
parser.add_argument(
    "--data_name",
    type=str,
    default="snli",
    choices=[
        "snli",
        "multi_nli",
        "anli-r1",
        "anli-r2",
        "anli-r3",
    ],
)
parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU.")
parser.add_argument(
    "--max_length", default=None, type=int, help="maximum sequence length."
)
# evaluation parameters
parser.add_argument("--perturbation", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
# pmp parameters
parser.add_argument(
    "--inference_method",
    type=str,
    default="analytic",
    choices=["standard", "analytic", "pmp"],
)
parser.add_argument("--control_regularization", type=float, default=0.001)
parser.add_argument("--control_scheme", type=str, default="P-I-D")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
if __name__ == "__main__":
    model_name = args.model_name
    pretrained_checkpoint = args.pretrained_checkpoint
    lora = args.lora
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha

    data_name = args.data_name
    batch_size = args.batch_size
    max_length = args.max_length

    perturbation = args.perturbation

    seed = args.seed

    inference_method = args.inference_method
    control_regularization = args.control_regularization
    control_scheme = args.control_scheme

    # For reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # set up gpus if available
    if torch.cuda.is_available():
        _gpu_ids = [i for i in range(torch.cuda.device_count())]
        _gpu_ids = [_gpu_ids[0]]
        batch_size = batch_size * len(_gpu_ids)
    else:
        _gpu_ids = []

    # construct model module for evaluation
    if (
        data_name == "snli"
        or data_name == "multi_nli"
        or data_name.startswith("anli")
        or data_name.startswith("adv_glue")
    ):
        num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    if model.config.model_type == "distilbert":
        target_model_modules = ["q_lin", "v_lin"]
    elif model.config.model_type == "bert":
        target_model_modules = ["query", "value"]
    elif model.config.model_type == "roberta":
        target_model_modules = ["query", "value"]
    elif model.config.model_type == "deberta-v2":
        target_model_modules = ["query_proj", "value_proj"]
    elif model.config.model_type == "opt":
        target_model_modules = ["q_proj", "v_proj"]
    else:
        raise ValueError(f"Unknown model. ")

    if lora == True:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_model_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    # load pre-trained weights if available
    if pretrained_checkpoint is not None:
        print("loading pre-trained model ...")
        pre_trained_weights_ = torch.load(pretrained_checkpoint)
        pre_trained_weights = OrderedDict()
        for key, value in pre_trained_weights_.items():
            if ".base_layer" in key:
                pre_trained_weights[key.replace(".base_layer", "")] = value
            else:
                pre_trained_weights[key] = value

        # pre_trained_weights = pre_trained_weights_
        
        model.load_state_dict(pre_trained_weights)

    if torch.cuda.is_available():
        model = model.cuda(_gpu_ids[0])
    elif torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()

    # construct data module
    dm_params = {
        "data_name": data_name,
        "batch_size": batch_size,
        "model_name": model_name,
        "max_padding": True,
        "perturbation": perturbation,
        "model": model,
        "pretrained_checkpoint": pretrained_checkpoint,
        "max_length": max_length,
    }
    dm = DataModule(**dm_params)

    if inference_method == "standard":
        pass
    elif inference_method == "analytic":
        control_config = {
            "target_modules": None,
            "batch_size": batch_size,
            "control_regularization": control_regularization,
            "control_scheme": control_scheme,
        }
        model = closed_loop_control_module(model, control_config, model_name)
        model.construct_embedding_module(
            model.embedding,
            model.sequential_model,
            model.classification,
            model.ccl_config,
            dm.train_dataloader(),
            dm.return_data_collator(),
        )
    elif inference_method == "pmp":
        control_config = {
            "target_modules": None,
            "batch_size": batch_size,
            "control_regularization": control_regularization,
            "control_scheme": control_scheme,
        }
        model = pmp_control_module(model, control_config, model_name)
        model.construct_embedding_module(
            model.embedding,
            model.sequential_model,
            model.classification,
            model.ccl_config,
            dm.train_dataloader(),
            dm.return_data_collator(),
        )

    # construct evaluation module
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        seed=seed,
        remove_unused_columns=False,  # this will keep input_ids and attention_mask in input
        label_names=["labels"],  # ccl model does not have associated labels defined
    )

    trainer = evaluator(
        model=model,
        train_dataset=dm.eval_dataloader(),
        eval_dataset=dm.eval_dataloader(),
        args=training_args,
        data_collator=dm.return_data_collator(),
    )

    start_time = time.time()
    eval_results = trainer.evaluate()
    end_time = time.time()
    wall_time = end_time - start_time
    print("The wall time of your code is:", wall_time)
        
    print("Evaluation accuracy:", eval_results["eval_loss"])
