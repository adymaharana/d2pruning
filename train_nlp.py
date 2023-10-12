# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""

import argparse
import json
import logging
import math
import os, pickle, sys
import random
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import torch

import datasets
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from core.training import Trainer, TrainingDynamicsLogger
from core.data import IndexDataset, CoresetSelection
from collections import Counter

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AdamW,
    # AutoConfig,
    # AutoModelForSequenceClassification,
    # AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification
)

# from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "imdb": ("text", None),
    "anli": ("premise", "hypothesis"),
    "yelp_review_full": ("text", None),
    "alisawuffles/WANLI": ("premise", "hypothesis"),
}

def preprocess_yelp(args, raw_datasets):
    df_train = pd.DataFrame(raw_datasets['train'])
    col_names = list(df_train.columns.values)

    if args.train_index_path:
        index = np.load(args.train_index_path)
        df_train = df_train.iloc[index]

    df_train.loc[df_train['label'] < 3, 'label'] = 0
    df_train.loc[df_train['label'] >= 3, 'label'] = 1

    for col in list(df_train.columns.values):
        if col not in col_names:
            df_train = df_train.drop(columns=[col])
    raw_datasets['train'] = datasets.Dataset.from_pandas(df_train).remove_columns(["__index_level_0__"])
    assert set(raw_datasets['train']['label']) == {0, 1}

    df_test = pd.DataFrame(raw_datasets['test'])
    col_names = list(df_test.columns.values)
    df_test.loc[df_test['label'] < 3, 'label'] = 0
    df_test.loc[df_test['label'] >= 3, 'label'] = 1

    for col in list(df_test.columns.values):
        if col not in col_names:
            df_test = df_test.drop(columns=[col])
    raw_datasets['test'] = datasets.Dataset.from_pandas(df_test)
    assert set(raw_datasets['test']['label']) == {0, 1}
    print(raw_datasets)
    return raw_datasets

def preprocess_wanli(args, raw_datasets):

    entailment_label_to_id = {"entailment": 0,
                              "neutral": 1,
                              "contradiction": 2}

    df_train = pd.DataFrame(raw_datasets['train'])
    col_names = list(df_train.columns.values)

    labels = [entailment_label_to_id[gold] for gold in df_train["gold"]]

    if os.path.exists(args.val_index_path):
        val_idxs = np.load(args.val_index_path).tolist()
    else:
        val_idxs = []
        label_counts = Counter(labels)
        for l, c in label_counts.items():
            val_count = int(c*5000/len(df_train))
            val_idxs += random.sample([idx for idx, label in enumerate(labels) if label == l], k=val_count)
        np.save(args.val_index_path, np.array(val_idxs))

    df_train["label"] = labels
    col_names += ["label"]
    for col in list(df_train.columns.values):
        if col not in col_names:
            df_train = df_train.drop(columns=[col])

    df_val = df_train.iloc[val_idxs]
    df_train = df_train.iloc[list(set(range(0, len(df_train))).difference(val_idxs))]

    raw_datasets['train'] = datasets.Dataset.from_pandas(df_train).remove_columns(["__index_level_0__"])
    raw_datasets['validation'] = datasets.Dataset.from_pandas(df_val).remove_columns(["__index_level_0__"])
    assert set(raw_datasets['train']['label']) == {0, 1, 2}

    df_test = pd.DataFrame(raw_datasets['test'])
    col_names = list(df_test.columns.values)
    labels = [entailment_label_to_id[gold] for gold in df_test["gold"]]
    df_test["label"] = labels
    col_names += ["label"]

    for col in list(df_test.columns.values):
        if col not in col_names:
            df_test = df_test.drop(columns=[col])
    raw_datasets['test'] = datasets.Dataset.from_pandas(df_test)
    assert set(raw_datasets['test']['label']) == {0, 1, 2}
    print(raw_datasets)
    return raw_datasets

def preprocess_counterfactual_imdb(args):
    data = pd.read_csv(args.eval_file, sep='\t')
    data["label"] = [0 if s.lower() == 'negative' else 1 for s in data["Sentiment"]]
    data["text"] = data["Text"]
    del data["Text"]
    dataset = datasets.Dataset.from_pandas(data)
    return DatasetDict({"test": dataset})


def preprocess_for_val(args, raw_datasets, val_size=5000, is_anli=False):

    if is_anli:
        df_train = pd.DataFrame(raw_datasets['train_r3'])
        df_test = pd.DataFrame(raw_datasets['test_r3'])
    else:
        df_train = pd.DataFrame(raw_datasets['train'])
        df_test = pd.DataFrame(raw_datasets['test'])

    total_num = len(df_train)

    if args.train_index_path:
        index = np.load(args.train_index_path)
        df_train = df_train.iloc[index]

    labels = df_train["label"]
    if os.path.exists(args.val_index_path):
        val_idxs = np.load(args.val_index_path).tolist()
        print("Loading validation set from saved index")
    else:
        val_idxs = []
        label_counts = Counter(labels)
        for l, c in label_counts.items():
            val_count = int(c*val_size/len(df_train))
            val_idxs += random.sample([idx for idx, label in enumerate(labels) if label == l], k=val_count)
        np.save(args.val_index_path, np.array(val_idxs))
        print("reating new index for validation set")

    df_val = df_train.iloc[val_idxs]
    df_train = df_train.iloc[list(set(range(0, len(df_train))).difference(val_idxs))]
    print("Reduced training set from %s to %s for creating validation set" % (total_num, len(df_train)))

    # raw_datasets['train'] = datasets.Dataset.from_pandas(df_train).remove_columns(["__index_level_0__"])
    if is_anli:
        raw_datasets['train_r3'] = datasets.Dataset.from_pandas(df_train).remove_columns(["__index_level_0__"])
        raw_datasets['validation_r3'] = datasets.Dataset.from_pandas(df_val).remove_columns(["__index_level_0__"])
    else:
        raw_datasets['train'] = datasets.Dataset.from_pandas(df_train).remove_columns(["__index_level_0__"])
        raw_datasets['validation'] = datasets.Dataset.from_pandas(df_val).remove_columns(["__index_level_0__"])

    if args.test_index_path:
        index = np.load(args.test_index_path)
        df_test = df_test.iloc[index]
        if is_anli:
            raw_datasets['test_r3'] = datasets.Dataset.from_pandas(df_test).remove_columns(["__index_level_0__"])
        else:
            raw_datasets['test'] = datasets.Dataset.from_pandas(df_test).remove_columns(["__index_level_0__"])

    return raw_datasets


"""Calculate td metrics"""
def EL2N(td_log, dataset, data_importance, num_labels, max_epoch=10):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        example = dataset[i]
        targets.append(example["labels"])
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_labels)
        el2n_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)

"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        example = dataset[i]
        targets.append(example["labels"])
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)
    data_importance['variance'] = []
    for i in range(data_size):
        data_importance['variance'].append([])

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        for i, idx in enumerate(index):
            data_importance['variance'][idx].append(target_prob[i].item())
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        record_training_dynamics(item)

    # compute variance
    sizes = [len(data_importance['variance'][i]) for i in range(data_size)]
    for i, s in enumerate(sizes):
        if s != sizes[0]:
            for j in range(sizes[0] - s):
                data_importance['variance'][i].append(1.0)
    data_importance['variance'] = torch.tensor(np.std(np.array(data_importance['variance']), axis=-1))
    # data_importance['variance'] = data_importance['variance'] - np.tile(np.expand_dims(np.mean(data_importance['variance'], axis=-1), axis=1), (1, sizes[0]))


class IndexedDataCollatorWithPadding(DataCollatorWithPadding):

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    # def __int__(self, tokenizer, padding, max_length, pad_to_multiple_of, return_tensors):
    #     super().__init__(tokenizer, padding, max_length, pad_to_multiple_of, return_tensors)

    def __call__(self, features):
        idxs = torch.tensor([idx for idx, f in features])
        text_features = [f for _, f in features]
        batch = self.tokenizer.pad(
            text_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return (idxs, batch)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="A seed for reproducible training.")
    parser.add_argument("--data_seed", type=int, default=0, help="A seed for reproducible pruning of data.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument("--early_stop_threshold", type=int, default=3, help="Number of epochs to wait before early stopping.")
    parser.add_argument("--early_stop", action="store_true", help="Whether to stop training early based on val dataset.")

    parser.add_argument("--do_train", action="store_true", help="Whether or not to train.")
    parser.add_argument("--do_eval", action="store_true", help="Whether or not to eval.")
    parser.add_argument("--do_test", action="store_true", help="Whether or not to eval on test.")
    parser.add_argument("--eval_train", action="store_true", help="Whether or not to evaluate the training dataset.")
    parser.add_argument("--eval_ood", action="store_true", help="Whether or not to evaluate the OOD dataset.")
    parser.add_argument("--ood_task", type=str)
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--save_confidence", action="store_true", help="save confidence values.")
    parser.add_argument("--save_feature", action="store_true", help="save feature emebddings.")
    parser.add_argument("--save_importance_scores", action="store_true", help="save confidence values.")
    parser.add_argument("--training_dynamics", action="store_true", help="save feature emebddings.")

    # additional arguments
    parser.add_argument(
        "--null_transform",
        action="store_true",
        help="Whether to train with empty strings as input for null-calculation required by V-information.",
    )
    parser.add_argument(
        "--train_logger",
        action="store_true",
        help="Log training dynamics"
    )

    ######################### Coreset Setting #########################
    parser.add_argument('--coreset', action='store_true', default=False)
    parser.add_argument('--coreset-mode', type=str, choices=['random', 'coreset', 'stratified', 'density', 'class', 'moderate'], default='random')
    parser.add_argument('--sampling-mode', type=str, choices=['kcenter', 'random', 'graph'], default='random')
    parser.add_argument('--budget-mode', type=str, choices=['uniform', 'density', 'confidence', 'aucpr'], default='random')

    parser.add_argument('--data-score-path', type=str)
    parser.add_argument('--bin-path', type=str)
    parser.add_argument('--feature-path', type=str)
    parser.add_argument('--coreset-key', type=str)
    parser.add_argument('--data-score-descending', type=int, default=0,
                        help='Set 1 to use larger score data first.')
    parser.add_argument('--class-balanced', type=int, default=0,
                        help='Set 1 to use the same class ratio as to the whole dataset.')
    parser.add_argument('--coreset-ratio', type=float)
    parser.add_argument('--label-balanced', action='store_true', default=False)

    ######################## Graph Sampling Setting ################################
    parser.add_argument('--n-neighbor', type=int, default=10)
    parser.add_argument('--median', action='store_true', default=False)

    #### Double-end Pruning Setting ####
    parser.add_argument('--mis-key', type=str)
    parser.add_argument('--mis-data-score-descending', type=int, default=0,
                        help='Set 1 to use larger score data first.')
    parser.add_argument('--mis-ratio', type=float)

    #### Reversed Sampling Setting ####
    parser.add_argument('--reversed-ratio', type=float,
                        help="Ratio for the coreset, not the whole dataset.")

    # Yelp
    parser.add_argument('--train-index-path', type=str)

    # Validation
    parser.add_argument('--val-index-path', type=str)
    parser.add_argument('--test-index-path', type=str)

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        # if args.push_to_hub:
        #     if args.hub_model_id is None:
        #         repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
        #     else:
        #         repo_name = args.hub_model_id
        #     repo = Repository(args.output_dir, clone_from=repo_name)
        #
        #     with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
        #         if "step_*" not in gitignore:
        #             gitignore.write("step_*\n")
        #         if "epoch_*" not in gitignore:
        #             gitignore.write("epoch_*\n")
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        # raw_datasets = load_dataset("glue", args.task_name)
        raw_datasets = load_dataset(args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # preprocess yelp
    if args.task_name == 'yelp_review_full':
        raw_datasets = preprocess_yelp(args, raw_datasets)
    elif args.task_name == 'alisawuffles/WANLI':
        raw_datasets = preprocess_wanli(args, raw_datasets)
    elif args.task_name in ['imdb']:
        raw_datasets = preprocess_for_val(args, raw_datasets, val_size=1000)
    elif args.task_name == 'anli':
        # raw_datasets = preprocess_for_val(args, raw_datasets, val_size=5000, is_anli = True)
        pass

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            if args.task_name == "anli":
                targets = raw_datasets["train_r3"]["label"]
                try:
                    label_list = raw_datasets["train_r3"].features["label"].names
                except AttributeError:
                    label_list = list(set(raw_datasets["train_r3"]["label"]))
            else:
                targets = raw_datasets["train"]["label"]
                try:
                    label_list = raw_datasets["train"].features["label"].names
                except AttributeError:
                    try:
                        label_list = list(set(raw_datasets["train"]["gold"]))
                    except KeyError:
                        label_list = list(set(raw_datasets["train"]["label"]))
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)


    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    # )

    # config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    # tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # model = BertForSequenceClassification.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    # )

    if args.coreset:
        df_train = pd.DataFrame(raw_datasets['train' if args.task_name != "anli" else "train_r3"])
        col_names = list(df_train.columns.values)
        total_num = len(df_train)

        if args.label_balanced:
            uniq_labels = list(set(targets))
            label_counts = Counter(targets)
            coreset_size_per_label = {k: int(v*args.coreset_ratio) for k, v in label_counts.items()}
            coreset_num = sum(list(coreset_size_per_label.values()))

        if args.coreset_key in ['entropy', 'forgetting', 'el2n', 'variance']:
            print("Using descending order")
            args.data_score_descending = 1

        score_index = None

        if args.coreset_mode == 'random':
            # random pruning
            if args.label_balanced:
                coreset_index = []
                for label in uniq_labels:
                    print("***************** Getting coresets for label", label, " ********************")
                    # get sample for the label
                    sample_idxs_by_label = torch.tensor(np.array([i for i in range(total_num) if targets[i] == label]))
                    coreset_index_by_label = CoresetSelection.random_selection(total_num=len(sample_idxs_by_label),
                                                                      num=coreset_size_per_label[label])
                    coreset_index.append(sample_idxs_by_label[coreset_index_by_label])
                coreset_index = torch.cat(coreset_index)

            else:
                coreset_index = CoresetSelection.random_selection(total_num=total_num,
                                                                  num=args.coreset_ratio * total_num)

        if args.coreset_mode == 'coreset':
            with open(args.data_score_path, 'rb') as f:
                data_score = pickle.load(f)
            coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key,
                                                                       ratio=args.coreset_ratio,
                                                                       descending=(args.data_score_descending == 1),
                                                                       class_balanced=args.label_balanced)

        if args.coreset_mode == 'moderate':
            with open(args.data_score_path, 'rb') as f:
                data_score = pickle.load(f)
            assert args.feature_path
            features = np.load(args.feature_path)
            coreset_index = CoresetSelection.moderate_selection(data_score=data_score, ratio=args.coreset_ratio, features=features)

        if args.coreset_mode == 'stratified':
            mis_num = int(args.mis_ratio * total_num)
            with open(args.data_score_path, 'rb') as f:
                data_score = pickle.load(f)
            data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin',
                                                                     mis_num=mis_num, mis_descending=False,
                                                                     coreset_key=args.coreset_key)

            coreset_num = int(args.coreset_ratio * total_num)
            # if args.sampling_mode != 'random' or args.budget_mode == 'aucpr':
            if True:
                assert args.feature_path
                features = np.load(args.feature_path)[score_index]
            else:
                features = None
            if args.budget_mode == 'confidence':
                data_score['confidence'] = data_score['confidence'][score_index]
            if args.label_balanced:
                coreset_index = []
                for label in uniq_labels:
                    print("***************** Getting coresets for label", label, " ********************")
                    # get sample for the label
                    sample_idxs_by_label = np.array([i for i, idx in enumerate(score_index) if targets[idx] == label])
                    data_score_by_label = {args.coreset_key: data_score[args.coreset_key][sample_idxs_by_label]}
                    if args.budget_mode == 'confidence':
                        data_score_by_label['confidence'] = data_score['confidence'][sample_idxs_by_label]
                    coreset_index_by_label, _ = CoresetSelection.stratified_sampling(data_score=data_score_by_label,
                                                                                     coreset_key=args.coreset_key,
                                                                                     coreset_num=coreset_size_per_label[label],
                                                                                     budget=args.budget_mode,
                                                                                     sampling=args.sampling_mode,
                                                                                     data_embeds=None if features is None else
                                                                                     features[sample_idxs_by_label],
                                                                                     n_neighbor=args.n_neighbor)
                    assert all(
                        [targets[idx] == label for idx in score_index[sample_idxs_by_label[coreset_index_by_label]]])
                    coreset_index = np.concatenate((coreset_index, sample_idxs_by_label[coreset_index_by_label]),
                                                   axis=0)
            else:
                coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score,
                                                                        coreset_key=args.coreset_key,
                                                                        coreset_num=coreset_num,
                                                                        budget=args.budget_mode,
                                                                        sampling=args.sampling_mode,
                                                                        data_embeds=features)
            coreset_index = score_index[coreset_index]

        if args.coreset_mode == 'density':
            mis_num = int(args.mis_ratio * total_num)
            with open(args.data_score_path, 'rb') as f:
                data_score = pickle.load(f)
            data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin',
                                                                     mis_num=mis_num, mis_descending=False,
                                                                     coreset_key=args.coreset_key)

            # bins = np.load(args.bin_path)[score_index]
            bins = np.load(args.bin_path)
            assert len(bins) == len(score_index)
            coreset_num = int(args.coreset_ratio * total_num)
            if True:
                assert args.feature_path
                features = np.load(args.feature_path)[score_index]
            else:
                features = None
            if args.budget_mode == 'confidence':
                data_score['confidence'] = data_score['confidence'][score_index]

            if args.label_balanced:
                coreset_index = []
                for label in uniq_labels:
                    print("***************** Getting coresets for label", label, " ********************")
                    # get sample for the label
                    sample_idxs_by_label = np.array([i for i, idx in enumerate(score_index) if targets[idx] == label])
                    data_score_by_label = {args.coreset_key: data_score[args.coreset_key][sample_idxs_by_label]}
                    if args.budget_mode == 'confidence':
                        data_score_by_label['confidence'] = data_score['confidence'][sample_idxs_by_label]
                    coreset_index_by_label, _ = CoresetSelection.density_sampling(data_score=data_score_by_label,
                                                                                  bins=bins[sample_idxs_by_label],
                                                                                  coreset_num=coreset_size_per_label[label],
                                                                                  budget=args.budget_mode,
                                                                                  sampling=args.sampling_mode,
                                                                                  data_embeds=None if features is None else
                                                                                  features[sample_idxs_by_label],
                                                                                  n_neighbor=args.n_neighbor)
                    assert all(
                        [targets[idx] == label for idx in score_index[sample_idxs_by_label[coreset_index_by_label]]])
                    coreset_index = np.concatenate((coreset_index, sample_idxs_by_label[coreset_index_by_label]),
                                                   axis=0)
            else:
                coreset_index, _ = CoresetSelection.density_sampling(data_score, bins=bins, coreset_num=coreset_num,
                                                                     budget=args.budget_mode,
                                                                     sampling=args.sampling_mode, data_embeds=features)
            coreset_index = score_index[coreset_index]

        if args.coreset_mode == 'class':
            mis_num = int(args.mis_ratio * total_num)
            with open(args.data_score_path, 'rb') as f:
                data_score = pickle.load(f)
            data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin',
                                                                     mis_num=mis_num, mis_descending=False,
                                                                     coreset_key=args.coreset_key)

            coreset_num = int(args.coreset_ratio * total_num)
            if True:
                assert args.feature_path
                features = np.load(args.feature_path)[score_index]
            else:
                features = None

            if args.budget_mode == 'confidence':
                data_score['confidence'] = data_score['confidence'][score_index]
            targets = np.array(targets)[score_index]
            coreset_index, _ = CoresetSelection.density_sampling(data_score, bins=targets, coreset_num=coreset_num,
                                                                 budget=args.budget_mode,
                                                                 sampling=args.sampling_mode, data_embeds=features,
                                                                 median=args.median)
            coreset_index = score_index[coreset_index]

        if len(coreset_index) < coreset_num:
            print(len(coreset_index), coreset_num)
            if score_index is not None:
                extra_sample_set = list(set(score_index.tolist()).difference(set(coreset_index.tolist())))
                coreset_index = np.hstack((coreset_index, np.array(random.sample(extra_sample_set,
                                                                                 k=min(len(extra_sample_set),
                                                                                       coreset_num - len(coreset_index))))))
            else:
                pass
            print(coreset_index.shape)

        df_train = df_train.iloc[coreset_index]
        n_pruned = len(df_train)

        logger.info(
            f"Pruned dataset from {total_num} samples to {n_pruned} samples, retaining {args.coreset_ratio} of the dataset."
        )
        print(col_names, list(df_train.columns.values))
        for col in list(df_train.columns.values):
            if col not in col_names:
                df_train = df_train.drop(columns=[col])
        raw_datasets['train' if args.task_name != "anli" else "train_r3"] = datasets.Dataset.from_pandas(df_train).remove_columns(["__index_level_0__"])
        print(raw_datasets)

        print("Pruned %s samples in original train set to %s" % (total_num, len(df_train)))
        task_name = args.task_name.split('/')[0]
        coreset_index_path = os.path.join(args.output_dir, f'coreset-{task_name}.npy')
        np.save(coreset_index_path, np.array(coreset_index))
    else:
        pass

    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression and False
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}


    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        if args.null_transform:
            n_samples = len(examples[sentence1_key])
            texts = (
                ([' ']*n_samples,) if sentence2_key is None else ([' ']*n_samples, [' ']*n_samples)
            )
        else:
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                print(label_to_id, examples["label"])
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets['train' if args.task_name != "anli" else "train_r3"].column_names,
            desc="Running tokenizer on dataset",
        )

    if args.task_name == 'anli':
        train_dataset = processed_datasets["train_r3"]
        eval_dataset = processed_datasets["dev_r3"]
        test_dataset = processed_datasets["test_r3"]
    else:
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
        test_dataset = processed_datasets["test"]

    if args.train_logger:
        train_dataset = IndexDataset(train_dataset)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        if args.train_logger:
            data_collator = IndexedDataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
        else:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))


    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)),
                                 batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset,
                                 collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)),
                                 batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    # if args.task_name is not None:
    #     # metric = load_metric("glue", args.task_name)
    #     metric = load_metric(args.task_name)
    # else:
    metric = load_metric("accuracy")
    best_metric = -1
    best_test_metric = -1
    plateau_counter = 0

    if args.train_logger:
        td_logger = TrainingDynamicsLogger()
    else:
        td_logger = None

    if args.do_train:

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                resume_step = None
                path = args.resume_from_checkpoint
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            if "epoch" in path:
                args.num_train_epochs -= int(path.replace("epoch_", ""))
            else:
                resume_step = int(path.replace("step_", ""))
                args.num_train_epochs -= resume_step // len(train_dataloader)
                resume_step = (args.num_train_epochs * len(train_dataloader)) - resume_step

        for epoch in range(args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            # for step, (idx, batch) in enumerate(train_dataloader):
            for step, batch in enumerate(train_dataloader):
                if args.train_logger:
                    (idx, batch) = batch
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == 0 and step < resume_step:
                    continue

                outputs = model(**batch)
                loss = outputs.loss

                if td_logger is not None:
                    log_tuple = {
                        'epoch': epoch,
                        'iteration': step,
                        'idx': idx.type(torch.long).clone(),
                        'output': F.log_softmax(outputs.logits, dim=1).detach().cpu().type(torch.half)
                    }
                    td_logger.log_tuple(log_tuple)

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: validation: {eval_metric}")

            save_checkpoint = False
            if eval_metric[metric.name] >= best_metric:
                best_metric = eval_metric[metric.name]
                save_checkpoint = True
                plateau_counter = 0
                if args.do_test:
                    for step, batch in enumerate(test_dataloader):
                        outputs = model(**batch)
                        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                        metric.add_batch(
                            predictions=accelerator.gather(predictions),
                            references=accelerator.gather(batch["labels"]),
                        )

                    test_metric = metric.compute()
                    best_test_metric = test_metric[metric.name]
                    logger.info(f"epoch {epoch}: test: {test_metric}")

            else:
                plateau_counter += 1

            if args.with_tracking:
                accelerator.log(
                    {
                        "accuracy" if args.task_name is not None else "glue": eval_metric,
                        "train_loss": total_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                )

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )

            if args.checkpointing_steps == "epoch" and save_checkpoint:
                # output_dir = f"epoch_{epoch}"
                # if args.output_dir is not None:
                #     output_dir = os.path.join(args.output_dir, output_dir)
                # accelerator.save_state(args.output_dir)

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    if args.push_to_hub:
                        repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            if args.early_stop and plateau_counter >= args.early_stop_threshold:
                break

        # if args.output_dir is not None:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(args.output_dir)
        #     if accelerator.is_main_process:
        #         tokenizer.save_pretrained(args.output_dir)
        #         if args.push_to_hub:
        #             repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        if args.task_name == "mnli":
            # Final evaluation on mismatched validation set
            eval_dataset = processed_datasets["validation_mismatched"]
            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
            )
            eval_dataloader = accelerator.prepare(eval_dataloader)

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

            eval_metric = metric.compute()
            logger.info(f"mnli-mm: {eval_metric}")

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "all_results.json"), "a+") as f:
                # json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)
                f.write(json.dumps(eval_metric) + '\n')
                if args.do_test:
                    f.write(json.dumps(test_metric) + '\n')
                f.write(json.dumps({'best_%s' % metric.name: best_metric}) + '\n')
                if args.do_test:
                    f.write(json.dumps({'best_test_%s' % metric.name: best_test_metric}) + '\n')

            if td_logger is not None:
                td_logger.save_training_dynamics(os.path.join(args.output_dir, 'td.pkl'), data_name=args.task_name)

    if args.do_eval:

        logger.info("***** Running evaluation *****")
        if args.eval_train:
            eval_dataloader = DataLoader(train_dataset, collate_fn=data_collator,
                                         batch_size=args.per_device_eval_batch_size, shuffle=False, drop_last=False)
            eval_dataloader = accelerator.prepare(eval_dataloader)
            logger.info(f"  Num examples = {len(train_dataset)}")
        elif args.eval_ood:
            assert args.ood_task
            if args.ood_task == 'cimdb':
                raw_datasets = preprocess_counterfactual_imdb(args)
            else:
                raw_datasets = load_dataset(args.ood_task)
            with accelerator.main_process_first():
                processed_datasets = raw_datasets.map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets['test' if args.ood_task != "anli" else "test_r3"].column_names,
                    desc="Running tokenizer on dataset",
                )
            eval_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator,
                                         batch_size=args.per_device_eval_batch_size, shuffle=False, drop_last=False)
            eval_dataloader = accelerator.prepare(eval_dataloader)
        else:
            logger.info(f"  Num examples = {len(eval_dataset)}")

        total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
        logger.info(f"  Total train batch size (w. parallel & distributed) = {total_batch_size}")

        confidences = []
        embeds = []
        losses = []
        entropies = []
        model.eval()
        for step, batch in tqdm(enumerate(eval_dataloader)):
            outputs = model(**batch, output_hidden_states=args.save_feature)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
            if args.save_confidence:
                if not is_regression:
                    probs = F.softmax(outputs.logits.detach().cpu(), dim=-1)
                    confs = [probs[k, l].item() for k, l in enumerate(batch['labels'])]
                    entropy = -1 * probs * torch.log(probs + 1e-10)
                    entropy = torch.sum(entropy, dim=1).detach().cpu().numpy().tolist()
                    entropies.extend(entropy)
                    loss = F.cross_entropy(outputs.logits, batch['labels'], reduction='none').detach().cpu().numpy().tolist()
                    losses.extend(loss)
                else:
                    preds = outputs.logits.squeeze()
                    confs = preds - batch["labels"]
                    confs = [abs(n) for n in confs.detach().cpu().numpy().tolist()]
                confidences.extend(confs)

            if args.save_feature:
                temp = outputs.hidden_states[-1]
                features = temp[:, 0, :].detach().cpu().numpy()
                embeds.append(features)

        eval_metric = metric.compute()
        logger.info(f"result: {eval_metric}")

        if args.save_confidence:
            if args.eval_train:
                out_file = os.path.join(args.output_dir, 'train_confs.npy')
            else:
                out_file = os.path.join(args.output_dir, 'eval_confs.npy')
            np.save(out_file, np.array(confidences))
            n_confs = len(confidences)
            logger.info(f"Saved confidence values for {n_confs} samples at {out_file}")

        if args.save_feature:
            if args.eval_train:
                out_file = os.path.join(args.output_dir, 'train-features.npy')
            else:
                out_file = os.path.join(args.output_dir, 'eval-features.npy')
            embeds = np.concatenate(embeds, axis=0)
            np.save(out_file, embeds)
            n_samples = embeds.shape[0]
            logger.info(f"Saved feature embeddings for {n_samples} samples at {out_file}")

        data_importance = {}
        if args.save_importance_scores:
            assert len(confidences) == len(entropies)
            assert len(entropies) == len(losses)
            data_importance['entropy'] = torch.zeros(len(entropies))
            data_importance['loss'] = torch.zeros(len(entropies))
            data_importance['confidence'] = torch.zeros(len(entropies))

            for i in range(len(confidences)):
                data_importance['entropy'][i] = entropies[i]
                data_importance['loss'][i] = losses[i]
                data_importance['confidence'][i] = confidences[i]

        if args.training_dynamics:
            td_path = os.path.join(args.output_dir, 'td.pkl')
            with open(td_path, 'rb') as f:
                pickled_data = pickle.load(f)
            training_dynamics = pickled_data['training_dynamics']
            training_dynamics_metrics(training_dynamics, train_dataset, data_importance)
            EL2N(training_dynamics, train_dataset, data_importance, num_labels, max_epoch=10)

            task_name = args.task_name.split('/')[0]
            data_score_path = os.path.join(args.output_dir, f'data-score-{task_name}.pickle')
            print(f'Saving data score at {data_score_path}')
            with open(data_score_path, 'wb') as handle:
                pickle.dump(data_importance, handle)

if __name__ == "__main__":
    main()
