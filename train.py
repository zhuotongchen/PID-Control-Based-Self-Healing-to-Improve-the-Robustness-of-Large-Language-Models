#!/usr/bin/env python3
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
import sys

sys.path.append("src")
from src.trainer_class import hf_module


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_name",
    type=str,
    default="distilbert-base-uncased",
    help="model configuration",
    choices=[
        "distilbert-base-uncased",
        "bert-large-uncased",
        "roberta-large",
        "roberta-base",
        "facebook/opt-1.3b",
    ],
)
parser.add_argument(
    "--pretrained_checkpoint", type=str, default=None, help="models are saved here"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./checkpoint/models",
    help="models are saved here",
)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
# data parameters
parser.add_argument(
    "--data_name",
    type=str,
    default="snli",
    choices=["snli", "multi_nli"],
)
parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU.")
parser.add_argument(
    "--gradient_accumulation_steps",
    default=1,
    type=int,
    help="gradient accumulation steps",
)
# training parameters
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
parser.add_argument("--seed", type=int, default=0)
# adversarial training parameters
parser.add_argument(
    "--perturbation",
    type=str,
    default=None,
    choices=["word_substitution-A2TYoo2021", "adv-pgd", "adv-freelb"],
)
parser.add_argument("--perturbation_ratio", type=float, default=0.2)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    opt = parser.parse_args()
    lm = hf_module(opt)
    lm.fit()
