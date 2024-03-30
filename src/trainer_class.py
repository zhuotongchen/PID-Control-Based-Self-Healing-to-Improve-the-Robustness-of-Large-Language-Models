import torch
import torch.nn as nn

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

from src.data_module import DataModule
from src.training_class import Trainer_standard
from adversarial_training_word_substitution import Trainer_adversarial_word_substitution
from adversarial_training_pgd import Trainer_adversarial_training

from peft import LoraConfig, get_peft_model
import os


class hf_module:
    def __init__(
        self,
        args,
        **kw,
    ):
        self.data_name = args.data_name
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.max_epochs = args.max_epochs
        self.pretrained_checkpoint = args.pretrained_checkpoint
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.perturbation = args.perturbation
        self.perturbation_ratio = args.perturbation_ratio
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.seed = args.seed

        if self.data_name == "snli" or self.data_name == "multi_nli":
            self.num_labels = 3
        else:
            raise ValueError(f"Unknown data name. ")

    def fit(self):
        # construct classifier
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels,
        )
            
        if self.model.config.model_type == "distilbert":
            target_model_modules = ["q_lin", "v_lin"]
        elif self.model.config.model_type == "bert":
            target_model_modules = ["query", "value"]
        elif self.model.config.model_type == "roberta":
            target_model_modules = ["query", "value"]
        elif self.model.config.model_type == "deberta-v2":
            target_model_modules = ["query_proj", "value_proj"]
        elif self.model.config.model_type == "opt":
            target_model_modules = ["q_proj", "v_proj"]
        else:
            raise ValueError(f"Unknown model. ")

        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_model_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

        # construct data module
        dm = DataModule(
            self.data_name,
            self.batch_size,
            self.model_name,
        )

        # load pre-trained weights if available
        if self.pretrained_checkpoint is not None:
            print("loading pre-trained model ...")
            pre_trained_weights = torch.load(self.pretrained_checkpoint)
            self.model.load_state_dict(pre_trained_weights)

        OUTPUT_DIR = (
            "data_{}_model_{}_lr_{}_perturbation_{}_augment_ratio_{}_Lora".format(
                self.data_name,
                self.model_name.replace("/", "_")
                if "/" in self.model_name
                else self.model_name,
                self.lr,
                self.perturbation if self.perturbation is not None else "none",
                self.perturbation_ratio,
            )
        )

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        training_args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.max_epochs,
            learning_rate=self.lr,
            fp16=False,
            save_total_limit=1,
            output_dir=OUTPUT_DIR,
            optim="adamw_hf",
            lr_scheduler_type="linear",
            warmup_ratio=0.05,
            report_to="tensorboard",
            logging_steps=1,
            evaluation_strategy="epoch",
            eval_steps=1,
            save_strategy="epoch",
            save_steps=1,
            logging_strategy="epoch",
            do_train=True,
            do_eval=True,
            seed=self.seed,
            remove_unused_columns=False,
        )
        if self.perturbation is None:
            trainer = Trainer_standard(
                model=self.model,
                train_dataset=dm.train_dataloader(),
                eval_dataset=dm.val_dataloader(),
                args=training_args,
                data_collator=dm.return_data_collator(),
            )
        elif self.perturbation.startswith("adv"):
            trainer = Trainer_adversarial_training(
                model=self.model,
                train_dataset=dm.train_dataloader(),
                eval_dataset=dm.val_dataloader(),
                args=training_args,
                data_collator=dm.return_data_collator(),
                tokenizer=dm.tokenizer,
                train_method=self.perturbation.split("-")[-1],
            )
        elif self.perturbation.startswith("word_substitution"):
            trainer = Trainer_adversarial_word_substitution(
                model=self.model,
                train_dataset=dm.train_dataloader(),
                eval_dataset=dm.val_dataloader(),
                args=training_args,
                data_collator=dm.return_data_collator(),
                tokenizer=dm.tokenizer,
                perturbation=self.perturbation.split("-")[-1],
                perturbation_ratio=self.perturbation_ratio,
                train_dataset_text=dm.train_original_dataset,
                data_fields=dm.key_names,
            )
        else:
            raise ValueError(f"Unknown perturbation type. ")

        trainer.train()
        self.model.config.use_cache = True
        torch.save(self.model.state_dict(), os.path.join(OUTPUT_DIR, "model_state.pth"))
