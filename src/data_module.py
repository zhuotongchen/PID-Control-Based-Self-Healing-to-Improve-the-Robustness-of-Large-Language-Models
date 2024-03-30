import torch
from torch.utils.data import Subset

from transformers import AutoTokenizer
from transformers import DefaultDataCollator, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict, Dataset
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import textattack

import pandas as pd
from tqdm import tqdm
import math
import random
from collections import OrderedDict
import os


class DataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_token_length=None):
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # implemented for batch size > 1
        input_ids = [t["input_ids"] for t in examples]
        lengths = [len(t) for t in input_ids]
        assert len(input_ids) == len(lengths)
        # Max for paddings
        if self.max_token_length is None:
            max_seq_len_ = max(lengths)
        else:
            max_seq_len_ = self.max_token_length

        pad_idx = self.tokenizer.unk_token_id
        tk_ = [torch.tensor(t + [pad_idx] * (max_seq_len_ - len(t))) for t in input_ids]
        return_masks = [
            torch.tensor([1] * len(t) + [0] * (max_seq_len_ - len(t)))
            for t in input_ids
        ]
        assert len(tk_) == len(input_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)
        tk_t = torch.stack(tk_, dim=0).long()
        return_masks = torch.stack(return_masks, dim=0).bool()
        
        if "deberta" in self.tokenizer.name_or_path:
            return_masks = return_masks.long()
            
        batch = {
            "input_ids": tk_t,
            "attention_mask": return_masks,
            "labels": torch.LongTensor([t["label"] for t in examples]),
        }
        return batch


class DataModule:
    """
    A module for handling and processing data for NLP tasks, specifically for models
    dealing with premises and hypotheses.

    Attributes:
        batch_size (int): The size of each batch of data.
        model_name (str): The name of the model for which the data is being prepared.
        tokenizer: Tokenizer object from the transformers library.
        tokenized_datasets: A dataset containing tokenized data.
        DataCollator: Collator object to handle data collation.
    """

    def __init__(
        self,
        data_name: str,
        batch_size: int,
        model_name: str,
        max_padding: bool = False,
        perturbation: str = None,
        model=None,
        pretrained_checkpoint=None,
        max_length=None,
        **kwargs,
    ):
        """
        Initialize the DataModule with the given parameters.

        Args:
            data_name (str): Name of the dataset.
            batch_size (int): Size of each data batch.
            model_name (str): Name of the model.
            max_padding (bool, optional): Flag to indicate if max padding is to be used. Defaults to False.
            perturbation (str, optional): Perturbation method, if any. Defaults to None.
            model: Model object, if available.
        """
        self.batch_size = batch_size
        self.model_name = model_name
        self.pretrained_checkpoint = pretrained_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._prepare_datasets(data_name, perturbation, model)

        if max_length is None:
            max_token_length = max(
                [len(example) for example in self.tokenized_datasets["train"]["input_ids"]]
            )
        else:
            max_token_length = max_length
        print("maximum token length:", max_token_length)

        self.DataCollator = DataCollator(
            self.tokenizer, max_token_length if max_padding == True else None
        )

    def _prepare_datasets(self, data_name: str, perturbation: str, model):
        """
        Prepares the datasets for training, validation, and evaluation.

        Args:
            data_name (str): The name of the dataset.
            perturbation (str): Perturbation method.
            model: The model object.
        """
        self.key_names = ["premise", "hypothesis"]

        if data_name in ["snli", "multi_nli"]:
            self._process_standard_datasets(data_name)
        elif data_name.startswith("anli"):
            self._process_anli_datasets(data_name)
        elif data_name.startswith("adv_glue"):
            self._process_adv_glue_datasets(data_name)

        tds = self._generate_datasets(
            self.train_original_dataset,
            self.key_names,
        )

        vds = self._generate_datasets(
            self.val_original_dataset,
            self.key_names,
        )

        if model is None and perturbation is None:
            pre_generated_data_dir = None
        elif self.pretrained_checkpoint is None:
            pre_generated_data_dir = os.path.join(
                "pre_generated_data",
                "data_name_{}_model_name_{}_perturbation_{}".format(
                    data_name.replace("-", "_"), model.config.model_type, perturbation
                ),
            )
        else:
            pre_generated_data_dir = os.path.join(
                "pre_generated_data",
                "model_name_{}_perturbation_{}".format(
                    self.pretrained_checkpoint.split("/")[0], perturbation
                ),
            )

        print(pre_generated_data_dir)

        eds = self._generate_datasets(
            dataset=self.eval_original_dataset,
            key_names=self.key_names,
            perturbation=perturbation,
            model=model,
            tokenizer=self.tokenizer,
            pre_generated_data_dir=pre_generated_data_dir,
        )

        ds = DatasetDict()
        if tds is not None:
            ds["train"] = tds
        if vds is not None:
            ds["val"] = vds
        if eds is not None:
            ds["eval"] = eds

        self.tokenized_datasets = ds.map(
            lambda x: self.tokenizer(x["input"]), remove_columns=["input"]
        )

    def _process_standard_datasets(self, data_name):
        dataset = load_dataset(data_name)
        dataset = dataset.filter(lambda x: x["label"] != -1)

        self.train_original_dataset = dataset["train"] if "train" in dataset else None
        
        self.train_original_dataset = self.train_original_dataset.shuffle(seed=42)
        self.train_original_dataset = self.train_original_dataset.select(range(10000))

        if "validation" in dataset:
            self.val_original_dataset = dataset["validation"]
        elif "validation_matched" in dataset:
            self.val_original_dataset = dataset["validation_matched"]
        else:
            self.val_original_dataset = None
        self.eval_original_dataset = (
            dataset["test"] if "test" in dataset else self.val_original_dataset
        )

    def _process_anli_datasets(self, data_name):
        dataset = load_dataset("anli")
        dataset = dataset.filter(lambda x: x["label"] != -1)
        suffix = data_name.split("-")[-1]
        self.train_original_dataset = dataset["train_" + suffix]
        self.val_original_dataset = dataset["dev_" + suffix]
        self.eval_original_dataset = dataset["test_" + suffix]
        
    def _process_adv_glue_datasets(self, dataset, data_name):
        suffix = data_name.split("-")[-1]
        self.train_original_dataset = None
        self.val_original_dataset = dataset.filter(lambda x: x["label"] != -1)
        self.eval_original_dataset = self.val_original_dataset

    def _generate_datasets(
        self,
        dataset,
        key_names,
        perturbation=None,
        model=None,
        tokenizer=None,
        pre_generated_data_dir=None,
    ):
        """
        Generates data based on the provided dataset, with optional perturbations.

        Args:
            dataset: The dataset to process.
            key_names (list of str): Key names to join for each data sample.
            perturbation (str, optional): Perturbation method. Defaults to None.
            model: Model used for generating adversarial examples. Required if perturbation is not None.
            tokenizer: Tokenizer used for generating adversarial examples. Required if perturbation is not None.
            pre_generated_data_dir (str, optional): Directory for pre-generated adversarial examples.

        Returns:
            Dataset: A dataset containing the processed data.
        """
        if dataset is None:
            return None

        if perturbation is None:
            data, labels = self._process_normal_data(dataset, key_names)
        else:
            data, labels = self._process_perturbed_data(
                dataset,
                key_names,
                perturbation,
                model,
                tokenizer,
                pre_generated_data_dir,
            )
        data_dict = {"input": data, "label": labels}
        data_df = pd.DataFrame.from_dict(data_dict)
        return Dataset.from_pandas(data_df)

    def _process_normal_data(self, dataset, key_names):
        """
        Processes normal data without perturbations.

        Args:
            dataset: The dataset to process.
            key_names (list of str): Key names to join for each data sample.

        Returns:
            tuple: Tuple containing data and labels.
        """
        data = [" ".join([sample[name] for name in key_names]) for sample in dataset]
        labels = dataset["label"]
        return data, labels

    def _process_perturbed_data(
        self, dataset, key_names, perturbation, model, tokenizer, pre_generated_data_dir
    ):
        """
        Processes data with perturbations.

        Args:
            dataset: The dataset to process.
            key_names (list of str): Key names to join for each data sample.
            perturbation (str): Perturbation method.
            model: Model used for generating adversarial examples.
            tokenizer: Tokenizer used for generating adversarial examples.
            pre_generated_data_dir (str): Directory for pre-generated adversarial examples.

        Returns:
            tuple: Tuple containing data and labels.
        """
        data_path = os.path.join(pre_generated_data_dir, "data.pth")
        labels_path = os.path.join(pre_generated_data_dir, "labels.pth")

        if os.path.exists(data_path) and os.path.exists(labels_path):
            print("Evaluation with pre-generated adversarial examples")
            return torch.load(data_path), torch.load(labels_path)

        assert model is not None and tokenizer is not None

        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer
        )
        build_attack = getattr(textattack.attack_recipes, perturbation)
        attack_generator = build_attack.build(model_wrapper=model_wrapper)

        num_total_samples = len(dataset)
        data, labels = [], []
        for step, sample in enumerate(dataset):
            if (step % math.ceil(num_total_samples * 0.1)) == 0:
                print(
                    "Construct augment dataset. Step:",
                    step,
                    "Total:",
                    num_total_samples,
                )

            example = OrderedDict({name: sample[name] for name in key_names})
            attack_result = attack_generator.attack(example, int(sample["label"]))
            if all(
                name in attack_result.perturbed_result.attacked_text._text_input
                for name in key_names
            ):
                X = " ".join(
                    [
                        attack_result.perturbed_result.attacked_text._text_input[name]
                        for name in key_names
                    ]
                )
            else:
                X = " ".join([sample[name] for name in key_names])
            data.append(X)
            labels.append(sample["label"])

        # save adversarial examples
        os.makedirs(pre_generated_data_dir)
        torch.save(data, os.path.join(pre_generated_data_dir, "data.pth"))
        torch.save(labels, os.path.join(pre_generated_data_dir, "labels.pth"))
        return data, labels

    def train_dataloader(self):
        return self.tokenized_datasets["train"]

    def val_dataloader(self):
        return self.tokenized_datasets["val"]

    def eval_dataloader(self):
        return self.tokenized_datasets["eval"]

    def return_data_collator(self):
        return self.DataCollator
