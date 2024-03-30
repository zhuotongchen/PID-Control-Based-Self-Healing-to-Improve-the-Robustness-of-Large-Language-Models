import math
import os
import random
import shutil
import sys
import time
from typing import Optional

from transformers.integrations import hp_params

import torch
from packaging import version
from torch.utils.data import DataLoader, RandomSampler

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    is_deepspeed_available,
)
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
    get_model_param_count,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    seed_worker,
    speed_metrics,
)
from transformers.utils import (
    is_accelerate_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

from src.training_class import Trainer_standard
import textattack
from collections import OrderedDict
import pandas as pd


class Trainer_adversarial_word_substitution(Trainer_standard):
    def __init__(
        self,
        train_dataset_text,
        perturbation,
        perturbation_ratio,
        data_fields,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dataset_text = train_dataset_text
        self.perturbation = perturbation
        self.perturbation_ratio = perturbation_ratio
        self.data_fields = data_fields

    def _get_augmented_train_sampler(
        self, train_dataset
    ) -> Optional[torch.utils.data.Sampler]:
        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.tokenizer.model_input_names[0]
                if self.tokenizer is not None
                else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(train_dataset)

    def get_adversarial_train_dataloader(
        self,
        model,
    ) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        ###### modification made for adversarial training
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, self.tokenizer
        )
        build_attack = getattr(textattack.attack_recipes, self.perturbation)
        adversarial_example_generator = build_attack.build(model_wrapper=model_wrapper)

        num_total_data = len(self.train_dataset_text)
        num_perturbed_data = int(num_total_data * self.perturbation_ratio)

        indices = list(range(num_total_data))
        random.shuffle(indices)
        count = 0
        adversarial_examples, labels = [], []
        for step, idx in enumerate(indices):
            if count > num_perturbed_data - 1:
                break

            if (count % math.ceil(num_perturbed_data * 0.02)) == 0:
                print(
                    "Number of augmented data:",
                    count,
                    "Number of searched data",
                    step,
                    "Total:",
                    num_perturbed_data,
                )

            data = self.train_dataset_text[idx]
            # generate adversarial example
            example = OrderedDict({name: data[name] for name in self.data_fields})
            attack_result = adversarial_example_generator.attack(
                example, int(data["label"])
            )
            # skip if the perturbed data is the same as the original data
            if all(
                attack_result.perturbed_result.attacked_text._text_input[name]
                == data[name]
                for name in self.data_fields
            ):
                continue
            # skip if the perturbed data does not have correct format
            if not all(
                name in attack_result.perturbed_result.attacked_text._text_input
                for name in self.data_fields
            ):
                continue
            # add this data into the dataset
            count += 1
            X = " ".join(
                [
                    attack_result.perturbed_result.attacked_text._text_input[name]
                    for name in self.data_fields
                ]
            )
            adversarial_examples.append(X)
            labels.append(data["label"])

        adversarial_examples_dict = {
            "input": adversarial_examples,
            "label": labels,
        }
        adversarial_examples_dict_df = pd.DataFrame.from_dict(adversarial_examples_dict)
        ds = datasets.Dataset.from_pandas(adversarial_examples_dict_df)
        tokenized_adversarial_examples = ds.map(
            lambda x: self.tokenizer(x["input"]), remove_columns=["input"]
        )
        concat_dataset = datasets.concatenate_datasets(
            [train_dataset, tokenized_adversarial_examples]
        )
        print("Number of concat dataset:", len(concat_dataset))
        if not isinstance(concat_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_augmented_train_sampler(
                concat_dataset
            )
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        #####################################
        return self.accelerator.prepare(DataLoader(concat_dataset, **dataloader_params))

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps)
                        * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader) * args.num_train_epochs
                    )
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps)
                    * args.gradient_accumulation_steps
                )
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            ###### modification made for adversarial training
            if self.perturbation is not None:
                epoch_iterator = self.get_adversarial_train_dataloader(model)
            else:
                epoch_iterator = train_dataloader
            #####################################

            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss, model, trial, epoch, ignore_keys_for_eval
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, model, trial, epoch, ignore_keys_for_eval
            )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)











    # def _inner_training_loop(
    #     self,
    #     batch_size=None,
    #     args=None,
    #     resume_from_checkpoint=None,
    #     trial=None,
    #     ignore_keys_for_eval=None,
    # ):
    #     self.accelerator.free_memory()
    #     self._train_batch_size = batch_size
    #     logger.debug(
    #         f"Currently training with a batch size of: {self._train_batch_size}"
    #     )

    #     model = self._wrap_model(self.model_wrapped)

    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader(model)

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = (
    #         self._train_batch_size * args.gradient_accumulation_steps * args.world_size
    #     )

    #     len_dataloader = None
    #     num_train_tokens = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = (
    #             len_dataloader // args.gradient_accumulation_steps
    #         )
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #         else:
    #             max_steps = math.ceil(
    #                 args.num_train_epochs * num_update_steps_per_epoch
    #             )
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = (
    #                 self.num_examples(train_dataloader) * args.num_train_epochs
    #             )
    #     elif (
    #         args.max_steps > 0
    #     ):  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torchrun or torch.distributed.launch (deprecated))."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    #     delay_optimizer_creation = False

    #     # We need to reset the scheduler, as its parameters may be different on subsequent calls
    #     if self._created_lr_scheduler:
    #         self.lr_scheduler = None
    #         self._created_lr_scheduler = False

    #     if self.is_deepspeed_enabled:
    #         self.optimizer, self.lr_scheduler = deepspeed_init(
    #             self, num_training_steps=max_steps
    #         )

    #     if not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None

    #     # Compute absolute values for logging, eval, and save if given as ratio
    #     if args.logging_steps is not None:
    #         if args.logging_steps < 1:
    #             self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
    #         else:
    #             self.state.logging_steps = args.logging_steps
    #     if args.eval_steps is not None:
    #         if args.eval_steps < 1:
    #             self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
    #         else:
    #             self.state.eval_steps = args.eval_steps
    #     if args.save_steps is not None:
    #         if args.save_steps < 1:
    #             self.state.save_steps = math.ceil(max_steps * args.save_steps)
    #         else:
    #             self.state.save_steps = args.save_steps

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         if args.gradient_checkpointing_kwargs is None:
    #             gradient_checkpointing_kwargs = {}
    #         else:
    #             gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

    #         self.model.gradient_checkpointing_enable(
    #             gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
    #         )

    #     # as the model is wrapped, don't use `accelerator.prepare`
    #     # this is for unhandled cases such as
    #     # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    #     use_accelerator_prepare = True if model is self.model else False

    #     if delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # prepare using `accelerator` prepare
    #     if use_accelerator_prepare:
    #         self.model.train()
    #         if hasattr(self.lr_scheduler, "step"):
    #             if self.use_apex:
    #                 model = self.accelerator.prepare(self.model)
    #             else:
    #                 model, self.optimizer = self.accelerator.prepare(
    #                     self.model, self.optimizer
    #                 )
    #         else:
    #             # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
    #             model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
    #                 self.model, self.optimizer, self.lr_scheduler
    #             )

    #     if self.is_fsdp_enabled:
    #         self.model = self.model_wrapped = model

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     # backward compatibility
    #     if self.is_deepspeed_enabled:
    #         self.deepspeed = self.model_wrapped

    #     # ckpt loading
    #     if resume_from_checkpoint is not None:
    #         if self.is_deepspeed_enabled:
    #             deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
    #         elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
    #             self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)

    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples:,}")
    #     logger.info(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(
    #         f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
    #     )
    #     if self.args.per_device_train_batch_size != self._train_batch_size:
    #         logger.info(
    #             f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
    #         )
    #     logger.info(
    #         f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
    #     )
    #     logger.info(
    #         f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    #     )
    #     logger.info(f"  Total optimization steps = {max_steps:,}")
    #     logger.info(
    #         f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
    #     )

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(
    #             os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #         )
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (
    #                 num_update_steps_per_epoch
    #             )
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info(
    #             "  Continuing training from checkpoint, will skip to saved global_step"
    #         )
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(
    #             f"  Continuing training from global step {self.state.global_step}"
    #         )
    #         if not args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first"
    #                 f" {steps_trained_in_current_epoch} batches in the first epoch."
    #             )

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     if self.hp_name is not None and self._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.state.trial_name = self.hp_name(self._trial)
    #     if trial is not None:
    #         assignments = (
    #             trial.assignments
    #             if self.hp_search_backend == HPSearchBackend.SIGOPT
    #             else trial
    #         )
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()

    #     self.control = self.callback_handler.on_train_begin(
    #         args, self.state, self.control
    #     )

    #     total_batched_samples = 0
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         ###### modification made for adversarial training
    #         if self.perturbation is not None:
    #             epoch_iterator = self.get_train_dataloader(model)
    #         else:
    #             epoch_iterator = train_dataloader
    #         #####################################
    #         if hasattr(epoch_iterator, "set_epoch"):
    #             epoch_iterator.set_epoch(epoch)

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = (
    #             len(epoch_iterator)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(
    #             args, self.state, self.control
    #         )

    #         if (
    #             epoch == epochs_trained
    #             and resume_from_checkpoint is not None
    #             and steps_trained_in_current_epoch == 0
    #         ):
    #             self._load_rng_state(resume_from_checkpoint)

    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if steps_trained_in_current_epoch > 0:
    #             epoch_iterator = skip_first_batches(
    #                 epoch_iterator, steps_trained_in_current_epoch
    #             )
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True

    #         step = -1
    #         for step, inputs in enumerate(epoch_iterator):
    #             total_batched_samples += 1

    #             if rng_to_sync:
    #                 self._load_rng_state(resume_from_checkpoint)
    #                 rng_to_sync = False

    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 if steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.update(1)
    #                 if steps_trained_in_current_epoch == 0:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                 continue
    #             elif steps_trained_progress_bar is not None:
    #                 steps_trained_progress_bar.close()
    #                 steps_trained_progress_bar = None

    #             if step % args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(
    #                     args, self.state, self.control
    #                 )

    #             with self.accelerator.accumulate(model):
    #                 tr_loss_step = self.training_step(model, inputs)

    #             if (
    #                 args.logging_nan_inf_filter
    #                 and not is_torch_tpu_available()
    #                 and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #             ):
    #                 # if loss is nan or inf simply add the average of previous logged losses
    #                 tr_loss += tr_loss / (
    #                     1 + self.state.global_step - self._globalstep_last_logged
    #                 )
    #             else:
    #                 tr_loss += tr_loss_step

    #             self.current_flos += float(self.floating_point_ops(inputs))

    #             is_last_step_and_steps_less_than_grad_acc = (
    #                 steps_in_epoch <= args.gradient_accumulation_steps
    #                 and (step + 1) == steps_in_epoch
    #             )

    #             if (
    #                 total_batched_samples % args.gradient_accumulation_steps == 0
    #                 or
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 is_last_step_and_steps_less_than_grad_acc
    #             ):
    #                 # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
    #                 # in accelerate. So, explicitly enable sync gradients to True in that case.
    #                 if is_last_step_and_steps_less_than_grad_acc:
    #                     self.accelerator.gradient_state._set_sync_gradients(True)

    #                 # Gradient clipping
    #                 if args.max_grad_norm is not None and args.max_grad_norm > 0:
    #                     # deepspeed does its own clipping
    #                     self.accelerator.clip_grad_norm_(
    #                         model.parameters(),
    #                         args.max_grad_norm,
    #                     )

    #                 # Optimizer step
    #                 self.optimizer.step()
    #                 optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
    #                 if optimizer_was_run:
    #                     # Delay optimizer scheduling until metrics are generated
    #                     if not isinstance(
    #                         self.lr_scheduler,
    #                         torch.optim.lr_scheduler.ReduceLROnPlateau,
    #                     ):
    #                         self.lr_scheduler.step()

    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = (
    #                     epoch + (step + 1 + steps_skipped) / steps_in_epoch
    #                 )
    #                 self.control = self.callback_handler.on_step_end(
    #                     args, self.state, self.control
    #                 )

    #                 self._maybe_log_save_evaluate(
    #                     tr_loss, model, trial, epoch, ignore_keys_for_eval
    #                 )
    #             else:
    #                 self.control = self.callback_handler.on_substep_end(
    #                     args, self.state, self.control
    #                 )

    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(
    #             args, self.state, self.control
    #         )
    #         self._maybe_log_save_evaluate(
    #             tr_loss, model, trial, epoch, ignore_keys_for_eval
    #         )

    #         if self.control.should_training_stop:
    #             break

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info(
    #         "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
    #     )
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     train_loss = self._total_loss_scalar / self.state.global_step

    #     metrics = speed_metrics(
    #         "train",
    #         start_time,
    #         num_samples=num_train_samples,
    #         num_steps=self.state.max_steps,
    #         num_tokens=num_train_tokens,
    #     )
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(
    #         use_mtime=False, output_dir=run_dir
    #     )

    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if (
    #         self.args.should_save
    #         and self.state.best_model_checkpoint is not None
    #         and self.args.save_total_limit == 1
    #     ):
    #         for checkpoint in checkpoints_sorted:
    #             if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
    #                 logger.info(
    #                     f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
    #                 )
    #                 shutil.rmtree(checkpoint)

    #     self.control = self.callback_handler.on_train_end(
    #         args, self.state, self.control
    #     )

    #     # Wait for the checkpoint to be uploaded.
    #     self._finish_current_push()

    #     # After training we make sure to retrieve back the original forward pass method
    #     # for the embedding layer by removing the forward post hook.
    #     if self.neftune_noise_alpha is not None:
    #         self._deactivate_neftune(self.model)

    #     return TrainOutput(self.state.global_step, train_loss, metrics)
