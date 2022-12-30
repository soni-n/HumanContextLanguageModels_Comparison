#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning HaRT for human language modeling.
"""

import logging
import math
import os
import sys

from args.mtl_args import DataTrainingArguments, ModelArguments
from src.model.mtl_eihart import MTL_EIHaRTPreTrainedModel
from src.model.modeling_hart import HaRTBaseLMHeadModel
from src.model.configuration_hart import HaRTConfig
from data.utils_hart.mtl_eihart_data_utils import load_dataset
from data.mtl_eihart_data_collator import DataCollatorWithPaddingForHaRT

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

# from HFtrainer_v451_for_reference.trainer import Trainer

logger = logging.getLogger(__name__)

class evalLogsCallback(TrainerCallback):

    def on_evaluate(self, args, state, control, **kwargs):
        if control.should_save:
            metrics = kwargs['metrics']
            # perplexity = math.exp(metrics["eval_loss"])
            # metrics["perplexity"] = perplexity
            self.save_metrics('eval_{}'.format(metrics['epoch']), metrics, args)

    def save_metrics(self, split, metrics, args):
        import json
        
        path = os.path.join(args.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "ac_task_name": data_args.ac_task_name,
        "ac_num_labels": data_args.ac_num_labels if data_args.ac_num_labels is not None else 1,
    }
    if model_args.instantiate_hart:
        config = HaRTConfig()
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    if model_args.add_history:
        config.add_history = True
        
    if model_args.use_qh05_wts:
        config.use_qh05_wts = True
    else:
        config.use_qh05_wts = False

    ## genralize this into initialization?
    print("Sum*****")
    config.ac_task = data_args.ac_task_name
    config.ac_num_labels = 1
    config.keys_to_ignore_at_inference += ["history", "last_hidden_state"]

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.instantiate_hart:
        def add_insep_token(tokenizer):
            special_tokens_dict = {'sep_token': str('<|insep|>')}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            assert num_added_toks == 1
            assert tokenizer.sep_token == '<|insep|>'
        
        add_insep_token(tokenizer)

    if model_args.instantiate_hart:
        hartbaseLMModel = HaRTBaseLMHeadModel.from_pretrained(model_args.model_name_or_path, config=config)
        hartbaseLMModel.resize_token_embeddings(len(tokenizer))
        model = MTL_EIHaRTPreTrainedModel(config, hartbaseLMModel)
    
    ###  commented out the following code snippet based on: https://discuss.huggingface.co/t/perplexity-from-fine-tuned-gpt2lmheadmodel-with-and-without-lm-head-as-a-parameter/16602
    # elif model_args.model_name_or_path and training_args.do_train: ## re-factor this code to use ArHuLM.from_pretrained, because lm_head is not being treated as a parameter
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_args.model_name_or_path,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    
    elif model_args.model_name_or_path:
        model = MTL_EIHaRTPreTrainedModel.from_pretrained(model_args.model_name_or_path)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
        model.resize_token_embeddings(len(tokenizer))

    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warn(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        
    #Dataset
    if data_args.train_table is not None or data_args.dev_table is not None or data_args.test_table is not None:
        if data_args.train_table is not None:
            train_data, train_uncut_blocks = load_dataset(logger, tokenizer, data_args.train_table, block_size, data_args.max_train_blocks, data_args, 'train', data_args.disable_hulm_batching)
        if data_args.dev_table is not None:
            eval_data, eval_uncut_blocks = load_dataset(logger, tokenizer, data_args.dev_table, block_size, data_args.max_val_blocks, data_args, 'dev', data_args.disable_hulm_batching)
        elif data_args.test_table is not None:
            eval_data, eval_uncut_blocks = load_dataset(logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', data_args.disable_hulm_batching)
    else:
        raise ValueError("This CLM runner requires mysql database tables as train/dev/test data sources currently!")

    train_dataset = train_data if training_args.do_train else None
    eval_dataset = eval_data

    def compute_metrics(p: EvalPrediction):
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        import scipy

        hulm_loss = p.predictions[1].mean().item()
        ac_loss = p.predictions[2].mean().item()
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) #if is_regression else np.argmax(preds, axis=-1)

        if True: # or is_regression:
            if data_args.save_preds_labels:
                np.savetxt(training_args.output_dir +'/preds.txt', preds)
                np.savetxt(training_args.output_dir + '/labels.txt', p.label_ids)
            mse = ((preds - p.label_ids) ** 2).mean().item()
            r_pear, p_value = scipy.stats.pearsonr(preds, p.label_ids)
            # from https://www.aclweb.org/anthology/W18-0604.pdf 
            r_meas1 = 0.77
            r_meas2 = 0.70
            r_dis = r_pear/((r_meas1*r_meas2)**0.5)

            return {
                'mse': mse,
                'r_dis': r_dis,
                'r_pear': r_pear,
                'p_value': p_value,
                'hulm_loss': hulm_loss,
                'ac_loss': ac_loss,
                'perplexity_hulm': math.exp(hulm_loss),
                }
        else:
            indices = p.label_ids!=-100 # make sure to ignore the labels marked as -100
            labels = p.label_ids[indices]
            if not model_args.use_hart_no_hist:
                preds = preds[indices]
            if data_args.save_preds_labels:
                np.savetxt(training_args.output_dir +'/preds.txt', preds, fmt='%d')
                np.savetxt(training_args.output_dir + '/labels.txt', labels, fmt='%d')
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

    # Data collator
    # This will take care of collating batches of type [users, windows, num_tokens]
    data_collator = DataCollatorWithPaddingForHaRT(model_args, config, tokenizer, training_args.deepspeed, is_user_level_ft=True) ## fix is_user_level_ft to generalize
    # data_collator = DataCollatorWithPaddingForHaRT(model_args, config, tokenizer, training_args.deepspeed) ## for debugging
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None, 
        eval_dataset=eval_dataset if training_args.do_eval or training_args.do_predict else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[evalLogsCallback] if training_args.do_train else None
    )  

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else: 
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["train_blocks_per_sample"] = train_uncut_blocks if data_args.max_train_blocks is None else min(data_args.max_train_blocks, train_uncut_blocks)
        metrics["block_size"] = block_size
        metrics["gpus"] = training_args.n_gpu
        metrics["total_epochs"] = training_args.num_train_epochs
        metrics["per_device_train_batch_size"] = training_args.per_device_train_batch_size
        metrics["train_table"] = data_args.train_table
        if model_args.instantiate_hart:
            metrics["history"] = model_args.add_history
            metrics["extract_layer"] = model_args.extract_layer if model_args.extract_layer else config.extract_layer if config.extract_layer else None
            metrics["layer_ins"] = model_args.layer_ins if model_args.layer_ins else config.layer_ins if config.layer_ins else None
            if model_args.add_history:
                metrics["0s_initial_history"] = False if model_args.initial_history else True
        

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        # perplexity = math.exp(metrics["eval_loss"])
        # metrics["perplexity"] = perplexity
        metrics["eval_blocks_per_sample"] = eval_uncut_blocks if data_args.max_val_blocks is None else min(data_args.max_val_blocks, eval_uncut_blocks) 
        metrics["per_device_eval_batch_size"] = training_args.per_device_eval_batch_size
        metrics["is_dev"] = True if data_args.dev_table else False
        metrics["eval_table"] = data_args.dev_table if data_args.dev_table else data_args.test_table

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Evaluation
    if training_args.do_predict:
        logger.info("*** Predict ***")

        eval_dataset, eval_uncut_blocks = load_dataset(logger, tokenizer, data_args.test_table, block_size, data_args.max_val_blocks, data_args, 'test', data_args.disable_hulm_batching)

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        # perplexity = math.exp(metrics["eval_loss"])
        # metrics["perplexity"] = perplexity
        metrics["eval_blocks_per_sample"] = eval_uncut_blocks if data_args.max_val_blocks is None else min(data_args.max_val_blocks, eval_uncut_blocks) 
        metrics["per_device_eval_batch_size"] = training_args.per_device_eval_batch_size
        metrics["is_dev"] = False
        metrics["eval_table"] = data_args.test_table

        trainer.log_metrics("eval_predict", metrics)
        trainer.save_metrics("eval_predict", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
