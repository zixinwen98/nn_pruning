from typing import Optional
import logging
from dataclasses import dataclass, field
from pathlib import Path
import random
import numpy as np
import copy
import os.path
import json
import timeit
import sys

from datasets import load_metric, load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    AutoModelForSequenceClassification,
    EvalPrediction,
    default_data_collator,
    AutoConfig,
    PretrainedConfig,
    set_seed
)
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import (
    HPSearchBackend,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)
from transformers.integrations import is_ray_available

from nn_pruning.hp_naming import TrialShortNamer


import shutil
from types import SimpleNamespace
import torch
import torch.cuda
import torch.nn as nn
from nn_pruning.inference_model_patcher import optimize_model
from nn_pruning.hp_naming import TrialShortNamer
from nn_pruning.patch_coordinator import SparseTrainingArguments
from nn_pruning.sparse_xp import SparseXP
from nn_pruning.sparse_trainer import SparseTrainer
if is_ray_available():
    from ray import tune

# from .glue_sparse_train import GlueSparseTrainer
import tempfile



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
}

logger = logging.getLogger(__name__)



class GlueTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        self.model_args = kwargs.pop("model_args")
        self.data_args = kwargs.pop("data_args")
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        data_args = self.data_args
        eval_dataset = self.additional_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]

        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.dataset_name]
        eval_datasets = [eval_dataset]
        if data_args.dataset_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(self.additional_datasets["validation_mismatched"])

        eval_dataloaders = []
        for eval_dataset in eval_datasets:
            eval_dataloaders.append( self.get_eval_dataloader(eval_dataset))

        # Temporarily disable metric computation, we will do it in the loop here.
        checkpoint_dir = self.checkpoint_dir()

        output0 = None
        for eval_dataloader, task in zip(eval_dataloaders, tasks):
            self.start_timer()

            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
            if output0 is None:
                output0 = output
            self.end_timer(len(eval_dataset), task)

            eval_results = output.metrics

            log_metrics = {f"eval_{task}_{k}": v for k, v in output.metrics.items()}
            self.log(log_metrics)

            output_eval_file = os.path.join(checkpoint_dir, f"eval_results_{task}.json")
            if self.is_world_process_zero():
                logger.info(f"***** Eval results {task} *****")
                for key, value in eval_results.items():
                    logger.info(f"  {key} = {value}")

                with open(output_eval_file, "w") as writer:
                    json.dump(eval_results, writer, indent=4, sort_keys=True)


        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.dataset_name]
        if data_args.dataset_name == "mnli":
            test_datasets = [self.additional_datasets["test_matched"]]
            tasks.append("mnli-mm")
            test_datasets.append(self.additional_datasets["test_mismatched"])
        else:
            test_datasets = [self.additional_datasets["test"]]

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            if "label" in test_dataset.column_names:
                test_dataset.remove_columns_("label")
            predictions = self.predict(test_dataset=test_dataset).predictions
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            predictions = np.squeeze(predictions) if self.is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(checkpoint_dir, f"test_results_{task}.tsv")
            if self.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if self.is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            if task in {"mnli", "mnli-mm", "rte", "wnli"}:
                                item = self.label_list[item]
                            writer.write(f"{index}\t{item}\n")

        super().finish_evaluate(checkpoint_dir, output0.metrics)

        return output0.metrics

    def checkpoint_dir(self):
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        checkpoint_dir = self.run_dir() / checkpoint_folder
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir()

        return checkpoint_dir

    def instrument_model(self, model):
        if self.args.optimize_model_before_eval != "disabled":
            model = optimize_model(self.model, self.args.optimize_model_before_eval)

        return TimingModule(model)

    def run_dir(self):
        # Save model checkpoint
        if hasattr(self, "_trial"):
            trial = self._trial
        else:
            trial = None
        if self.hp_search_backend is not None and trial is not None:
            run_id = trial.number if self.hp_search_backend == HPSearchBackend.OPTUNA else tune.get_trial_id()
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = Path(self.args.output_dir) / run_name
        else:
            run_dir = Path(self.args.output_dir)

        return run_dir

    def start_timer(self):
        self._model_save = self.model
        self.model = self.instrument_model(self.model)
        self._start_time = timeit.default_timer()

    def end_timer(self, eval_dataset_length, suffix = None):
        evalTime = timeit.default_timer() - self._start_time
        cudaEvalTime, cudaEvalCount = self.model.get_results()
        cudaEvalTime = 1e-3 * cudaEvalTime
        checkpoint_dir = self.checkpoint_dir()
        suffix = "" if suffix is None else "_" + suffix
        timing_file = os.path.join(checkpoint_dir, f"evaluate_timing{suffix}.json")

        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / eval_dataset_length)
        logger.info("  Cuda time %f secs (%f sec per example)", cudaEvalTime, cudaEvalTime / eval_dataset_length)

        with open(timing_file, "w") as f:
            f.write(json.dumps({"eval_elapsed_time": evalTime, "cuda_eval_elapsed_time": cudaEvalTime}))

        self.model = self._model_save


    def finish_evaluate(self, checkpoint_dir, metrics):
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        for k, v in self.__dict__.items():
            if k.endswith("_args") and k != "args":
                filename = k + ".json"
                s = json.dumps(v.__dict__, indent=4, sort_keys=True)
                with open(os.path.join(checkpoint_dir, filename), "w") as f:
                    f.write(s)

# SparseTrainer should appear first in the base classes, as its functions must override QATrainer and its base classes (Trainer)
class GlueSparseTrainer(SparseTrainer, GlueTrainer):
    def __init__(self, sparse_args, *args, **kwargs):
        GlueTrainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)


class SparseGlueShortNamer(TrialShortNamer):
    DEFAULTS = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-08,
        "ampere_pruning_method": "disabled",
        "attention_block_cols": 32,
        "attention_block_rows": 32,
        "attention_lambda": 1.0,
        "attention_output_with_dense": 0,
        "attention_pruning_method": "sigmoied_threshold",
        "bias_mask": True,
        "dataloader_drop_last": False,
        "dataloader_num_workers": 0,
        "dataset_cache_dir": "dataset_cache_dir",
        "debug": False,
        "dense_block_cols": 1,
        "dense_block_rows": 1,
        "dense_lambda": 1.0,
        "dense_pruning_method": "sigmoied_threshold:1d_alt",
        "disable_tqdm": False,
        "distil_alpha_ce": 0.1,
        "distil_alpha_teacher": 0.9,
        "distil_teacher_name_or_path": "aloxatel/bert-base-mnli",
        "distil_temperature": 2.0,
        "do_eval": 1,
        "do_predict": False,
        "do_train": 1,
        "doc_stride": 128,
        "eval_steps": 5000,
        "evaluation_strategy": "steps",
        "final_ampere_temperature": 20.0,
        "final_finetune": False,
        "final_threshold": 0.1,
        "final_warmup": 5,
        "fp16": False,
        "fp16_opt_level": "O1",
        "gradient_accumulation_steps": 1,
        "ignore_data_skip": False,
        "initial_ampere_temperature": 0.0,
        "initial_threshold": 0,
        "initial_warmup": 1,
        "learning_rate": 3e-05,
        "load_best_model_at_end": False,
        "local_rank": -1,
        "logging_first_step": False,
        "logging_steps": 250,
        "mask_init": "constant",
        "mask_scale": 0.0,
        "mask_scores_learning_rate": 0.01,
        "max_grad_norm": 1.0,
        "max_seq_length": 128,
        "max_steps": -1,
        "model_name_or_path": "bert-base-uncased",
        "model_parallel": False,
        "no_cuda": False,
        "num_train_epochs": 10,
        "optimize_model_before_eval": "disabled",
        "output_dir": "output/mnli_test/",
        "overwrite_cache": 0,
        "overwrite_output_dir": 1,
        "pad_to_max_length": True,
        "past_index": -1,
        "per_device_eval_batch_size": 8,
        "per_device_train_batch_size": 1,
        "prediction_loss_only": False,
        "regularization": "l1",
        "regularization_final_lambda": 10,
        "remove_unused_columns": True,
        "run_name": "output/mnli_test/",
        "save_steps": 5000,
        "save_total_limit": 50,
        "seed": 17,
        "dataset_name": "mnli",
        "tpu_metrics_debug": False,
        "use_fast_tokenizer": True,
        "warmup_steps": 5400,
        "weight_decay": 0.0,
        'fp16_backend': 'auto',
        'sharded_ddp': False,
        'layer_norm_patch': False,
        'layer_norm_patch_steps': 50000,
        'layer_norm_patch_start_delta': 0.99,
        'gelu_patch': False,
        'gelu_patch_steps': 50000,
        'eval_with_current_patch_params': False,
        'warmup_ratio': 0.0,
        'fp16_full_eval': False,
        'label_smoothing_factor': 0.0,
        'adafactor': False,
        'group_by_length': False,
        'report_to': [],
        'dataloader_pin_memory': True,
        'skip_memory_metrics': False,
        '_n_gpu': 1,
        'linear_min_parameters': 0.005,
        'lr_scheduler_type': 'SchedulerType.LINEAR',
        'logging_strategy': 'IntervalStrategy.STEPS',
        'save_strategy': 'IntervalStrategy.STEPS',
        'rewind_model_name_or_path': None,
        'qat': False,
        'qconfig': 'default',
    }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    dataset_cache_dir: Optional[str] = field(
        default="dataset_cache", metadata={"help": "The path to the dataset cache."}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."


@dataclass
class GlueDataTrainingArguments(DataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            self.dataset_name = self.dataset_name.lower()
            if self.dataset_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."


@dataclass
class PruningTrainingArguments(TrainingArguments):
    optimize_model_before_eval: str = field(
        default="disabled",
        metadata={
            "help": "Apply some optimization to model before evaluation (use nn_pruning.inference_model_patcher.InferencePatcher)."
                    "Valid values: disabled, block_sparse, dense"
        },
    )


class Pruning:
    ARGUMENTS = {
        "model": ModelArguments,
        "data": DataTrainingArguments,
        "training": PruningTrainingArguments,
    }

    def __init__(self, param_dict):
        # See all possible arguments in src/transformers/training_args.py
        # or by passing the --help flag to this script.
        # We now keep distinct sets of args, for a cleaner separation of concerns.
        arguments = copy.deepcopy(self.ARGUMENTS)
        self.arguments_names = list(arguments.keys())
        parser = HfArgumentParser(arguments.values())
        parse_results = parser.parse_dict(param_dict) #, strict=True)

        assert self.arguments_names[0] == "model"
        assert self.arguments_names[1] == "data"
        assert self.arguments_names[2] == "training"

        # Explicitly affect args, to make IDE not flagging members as unknown
        self.model_args = parse_results[0]
        self.data_args = parse_results[1]
        self.training_args = parse_results[2]

        for i, (k, v) in enumerate(arguments.items()):
            if i < 3:
                continue
            setattr(self, k + "_args", parse_results[i])

    def model_init(self, trial=None):
        model =  self._model_init(self.model_args, self.config, self.data_args)
        if hasattr(model.config, "layer_norm_type") and model.config.layer_norm_type == "no_norm":
            from nn_pruning.modules.nonorm import NoNormPatcher
            nnc = NoNormPatcher()
            nnc.patch(model)

        return model


    def get_all_args(self, exclude_base=False):
        # Extract the other arguments
        all_args = {}
        for k in self.arguments_names:
            if exclude_base and k == "training":
                continue
            name = k + "_args"
            all_args[name] = getattr(self, name)
        return all_args
    
    @classmethod
    def run_from_dict(cls, param_dict):
        r = cls(param_dict)
        return r.run()
    '''
    @classmethod
    def run_from_json_file(cls, filename):
        json_file_name = Path(filename).resolve()
        param_dict = json.load(open(json_file_name))
        return cls.run_from_dict(param_dict)

    @classmethod
    def run_from_command_line(cls):
        if len(sys.argv) < 2:
            raise RuntimeError("Please specify json file")
        cls.run_from_json_file(sys.argv[1])
    '''
    def setup_logging(self):
        # Setup logging
        training_args = self.training_args
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    def create_directories(self):
        training_args = self.training_args
        output_dir = Path(training_args.output_dir).resolve()

        if (
            output_dir.exists()
            and list(output_dir.iterdir())
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )

    def initial_message(self):
        # Log on each process the small summary:
        training_args = self.training_args
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()

        logger.info("Training/evaluation parameters")
        for k in self.arguments_names:
            logger.info("  %s: %s", k.capitalize(), getattr(self, k + "_args"))

    def setup_random(self):
        # Set seed before initializing model.
        training_args = self.training_args
        set_seed(training_args.seed)


    def create_tokenizer(self):
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.
        model_args = self.model_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
        )

        # Tokenizer check: this script requires a fast tokenizer.
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
                "requirement"
            )

    def create_trainer(self):
        self.trainer = None
        raise RuntimeError("Implement in subclass")

    def prepare(self):
        self.create_directories()
        self.setup_logging()
        self.initial_message()
        self.setup_random()
        self.create_tokenizer()
        self.create_dataset()
        self.create_config()
        self.create_trainer()

    def train(self):
        # Training
        self.trainer.train()
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

    def evaluate(self):
        logger.info("*** Evaluate ***")
        _ = self.trainer.evaluate()

    def run(self):
        self.prepare()

        if self.training_args.do_train:
            self.train()

        # Evaluation
        results = {}
        if self.training_args.do_eval:
            self.evaluate()

        return results

    def hp_name(self, trial):
        all_args = self.get_all_args()
        d = {}
        for key, value in all_args.items():
            for k, v in value.__dict__.items():
                if v is None:
                    continue
                try:
                    _ = json.dumps(v)
                except Exception as e:
                    if k.startswith("__"):
                        continue
                    else:
                        v = str(v)
                        if k == "evaluation_strategy":
                            v = v.split(".")[-1].lower()
                    print(k, v)

                if k in d:
                    raise RuntimeError(f"Duplicate parameters {k} in arguments")
                d[k] = v

        EXCLUDES = ["logging_dir"]
        for exclude in EXCLUDES:
            del d[exclude]
        try:
            ret = self.SHORT_NAMER.shortname(d)
        except:
            raise
        return ret

    def hyperparameter_search(self, direction="maximize", hp_space=None, n_trials=1):
        self.prepare()

        def default_hp_space_fun(trial):
            return {}

        if hp_space is None:
            hp_space = default_hp_space_fun

        return self.trainer.hyperparameter_search(
            hp_name=self.hp_name,
            direction=direction,
            hp_space=hp_space,
            n_trials=n_trials,
        )

    @classmethod
    def fix_last_checkpoint_bug_checkpoint(cls, checkpoint_path):
        # Special stuff : add link to compensate for bug
        for link_name in ["pytorch_model.bin", "training_args.bin", "vocab.txt", "tokenizer_config.json",
                          "special_tokens_map.json"]:
            filename = checkpoint_path / link_name
            print(filename)
            filename_parent = Path("..") / link_name
            filename_absolute_parent = checkpoint_path.parent / link_name
            if not filename.exists() and filename_absolute_parent.exists():
                print(filename, filename_parent)
                filename.symlink_to(filename_parent)

    @classmethod
    def fix_last_checkpoint_bug(cls, run_path):
        run_path = Path(run_path)
        for src_path in run_path.iterdir():
            if src_path.name.startswith("checkpoint-"):
                cls.fix_last_checkpoint_bug_checkpoint(src_path)


class GluePruning(Pruning):
    
    ARGUMENTS = {
        "model": ModelArguments,
        "data": GlueDataTrainingArguments,
        "training": PruningTrainingArguments,
    }
    GLUE_TRAINER_CLASS = GlueTrainer
    SHORT_NAMER = TrialShortNamer

    @classmethod
    def _model_init(cls, model_args, model_config, data_args):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=model_config,
            cache_dir=model_args.cache_dir,
        )
        return model

    def create_config(self):
        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        model_args = self.model_args

        if hasattr(self, "patch_coordinator"):
            teacher = self.patch_coordinator.teacher
        else:
            teacher = None

        config_path = model_args.config_name if model_args.config_name else model_args.model_name_or_path
        if teacher is not None:
            id2label = teacher.config.id2label
            label2id = {v: k for k, v in id2label.items()}
            kwargs = dict(id2label=id2label, label2id=label2id)
        else:
            config = AutoConfig.from_pretrained(config_path)
            kwargs = dict(id2label=config.id2label, label2id=config.label2id)

        self.config = AutoConfig.from_pretrained(
            config_path,
            num_labels=self.num_labels,
            finetuning_task=self.data_args.dataset_name,
            cache_dir=model_args.cache_dir,
            **kwargs
        )

        return self.config

    def create_dataset(self):
        # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
        # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
        # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
        # label if at least two columns are provided.
        #
        # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
        # single column. You can easily tweak this behavior (see below)
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        data_args = self.data_args
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset("glue", data_args.dataset_name)
        elif data_args.train_file.endswith(".json"):
            # Loading a dataset from local json files
            datasets = load_dataset(
                "json",
                data_files={
                    "train": data_args.train_file,
                    "validation": data_args.validation_file,
                },
            )
        else:
            raise RuntimeError("Please use either json or dataset name")


        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Labels
        if data_args.dataset_name is not None:
            is_regression = data_args.dataset_name == "stsb"
            if not is_regression:
                label_list = datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                label_list = None
                num_labels = 1
        else:
            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_regression = datasets["train"].features["label"].dtype in [
                "float32",
                "float64",
            ]
            if is_regression:
                num_labels = 1
            else:
                # A useful fast method:
                # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                label_list = datasets["train"].unique("label")
                label_list.sort()  # Let's sort it for determinism
                num_labels = len(label_list)
        self.is_regression = is_regression
        self.label_list  = label_list
        self.num_labels = num_labels

        # Preprocessing the datasets
        if data_args.dataset_name is not None:
            sentence1_key, sentence2_key = task_to_keys[data_args.dataset_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
            max_length = data_args.max_seq_length
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False
            max_length = None

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None

        model_config = self.create_config()

        if (
                model_config.label2id != PretrainedConfig(num_labels=num_labels).label2id
                and data_args.dataset_name is not None
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model_config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
            else:
                logger.warn(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif data_args.dataset_name is None:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        self.label_to_id = label_to_id

        # This is needed because some part of the dataset contains label = -1
        if label_to_id is not None:
            preprocess_label_to_id = copy.deepcopy(label_to_id)
            preprocess_label_to_id[-1] = -1
        else:
            preprocess_label_to_id = None

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if preprocess_label_to_id is not None and "label" in examples:
                result["label"] = [preprocess_label_to_id[l] for l in examples["label"]]
            return result

        cache_file_names = {}
        cache_dir = (Path(data_args.dataset_cache_dir) / data_args.dataset_name).resolve()
        cache_dir.mkdir(exist_ok=True, parents=True)
        for key in ["train"]:
            cache_file_names[key] =  str(cache_dir / key)

        for key in ["validation", "test"]:
            if data_args.dataset_name == "mnli":
                for matched in ["matched", "mismatched"]:
                    key_matched = "_".join([key, matched])
                    cache_file_names[key_matched] = str(cache_dir / key_matched)
            else:
                cache_file_names[key] = str(cache_dir / key)


        datasets = datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            cache_file_names = cache_file_names
        )
        self.datasets = datasets

        self.train_dataset = datasets["train"]
        self.eval_dataset = datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
        if data_args.dataset_name is not None:
            self.test_dataset = datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
        # Log a few random samples from the training set:
        for index in random.sample(range(len(self.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {self.train_dataset[index]}.")


    def create_trainer(self):
        # Data collator
        # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
        # collator.
        training_args = self.training_args
        data_args = self.data_args

        # Get the metric function
        if data_args.dataset_name is not None:
            metric = load_metric("glue", data_args.dataset_name)

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
            if data_args.dataset_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif self.is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


        all_args = self.get_all_args(exclude_base=True)

        # Initialize our Trainer
        self.trainer = self.GLUE_TRAINER_CLASS(
            model=None,
            args=training_args,
            train_dataset=self.train_dataset if training_args.do_train else None,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=default_data_collator if data_args.pad_to_max_length else None,
            model_init=self.model_init,
            **all_args,
        )
        self.trainer.is_regression = self.is_regression
        self.trainer.label_list = self.label_list

        self.trainer.additional_datasets = self.datasets

    @classmethod
    def evaluate_model(cls, model_name_or_path, task, optimize_mode="dense", output_dir = None):
        if output_dir is None:
            output_dir = Path(model_name_or_path)
        else:
            output_dir = Path(output_dir)
        output_dir = output_dir.resolve()

        parameters = {
            "model_name_or_path": str(model_name_or_path),
            "dataset_name": task,
            "dataset_cache_dir": "dataset_cache_dir",
            "do_train": 0,
            "do_eval": 1,
            "per_device_eval_batch_size": 128,
            "max_seq_length": 128,
            "doc_stride": 128,
            "output_dir": str(output_dir),
            "logging_dir": str(output_dir),
            "overwrite_cache": 0,
            "overwrite_output_dir": 0,
            "optimize_model_before_eval": optimize_mode
        }

        cls.run_from_dict(parameters)

        file_info = {"timings": "evaluate_timing",
                     "metrics": "eval_results"}

        if task is not None:
            file_info_tmp = {}
            for k, v in file_info.items():
                file_info_tmp[k] = file_info[k] + "_" + task
                if task == "mnli":
                    file_info_tmp[k + "_mm"] = file_info_tmp[k] + "-mm"
            file_info = file_info_tmp

        ret = {}
        for k, v in file_info.items():
            with open(output_dir / "checkpoint-0" / (v + ".json")) as f:
                j = json.load(f)
                ret[k] = j

        return ret

class GlueSparsePruning(SparseXP, GluePruning):
    ARGUMENTS = {
        "model": ModelArguments,
        "data": GlueDataTrainingArguments,
        "training": PruningTrainingArguments,
        "sparse": SparseTrainingArguments,
    }
    GLUE_TRAINER_CLASS = GlueSparseTrainer
    SHORT_NAMER = SparseGlueShortNamer
    CONSTRUCTOR = AutoModelForSequenceClassification
    LOGIT_NAMES = ["logits"]

    def __init__(self, params):
        GluePruning.__init__(self, params)
        SparseXP.__init__(self)

    def create_trainer(self, *args, **kwargs):
        super().create_trainer(*args, **kwargs)
        SparseXP.setup_trainer(self)

    @classmethod
    def final_finetune(cls, src_path, dest_path, task, teacher):
        param_dict = {
            "model_name_or_path": src_path,
            "dataset_name": task,
            "dataset_cache_dir": "dataset_cache_dir",
            "do_train": 1,
            "do_eval": 1,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 128,
            "max_seq_length": 128,
            "doc_stride": 128,
            "num_train_epochs": 6,
            "logging_steps": 250,
            "save_steps": 5000,
            "eval_steps": 5000,
            "save_total_limit": 50,
            "seed": 17,
            "evaluation_strategy": "steps",
            "learning_rate": 3e-5,
            "output_dir": dest_path,
            "logging_dir": dest_path,
            "overwrite_cache": 0,
            "overwrite_output_dir": 1,
            "warmup_steps": 10,
            "initial_warmup": 0,
            "final_warmup": 0,
            "mask_init": "constant",
            "mask_scale": 0.0,
            "regularization": "",
            "regularization_final_lambda": 0,
            "distil_teacher_name_or_path":teacher,
            "distil_alpha_ce": 0.1,
            "distil_alpha_teacher": 0.90,
            "attention_output_with_dense": 0,
            "final_finetune": 1,
        }


        glue = cls(param_dict)
        glue.run()

        cls.fix_last_checkpoint_bug(dest_path)


def main():
    param_dict = {
            "model_name_or_path": "roberta-base",
            "dataset_name": "mnli",
            "dataset_cache_dir": "dataset_cache_dir",
            "do_train": 1,
            "do_eval": 1,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 128,
            "max_seq_length": 128,
            "doc_stride": 128,
            "num_train_epochs": 12,
            "logging_steps": 250,
            "save_steps": 5000,
            "eval_steps": 5000,
            "save_total_limit": 50,
            "seed": 17,
            "evaluation_strategy": "steps",
            "learning_rate": 3e-5,
            "mask_scores_learning_rate": 1e-2,
            "output_dir": "output/mnli_test2/",
            "logging_dir": "output/mnli_test2/",
            "overwrite_cache": 0,
            "overwrite_output_dir": 1,
            "warmup_steps": 12000,
            "initial_warmup": 1,
            "final_warmup": 4,
            "initial_threshold": 0,
            "final_threshold": 0.1,
            "dense_pruning_method": "sigmoied_threshold:1d_alt",
            "dense_block_rows":1,
            "dense_block_cols":1,
            "dense_lambda":1.0,
            "attention_pruning_method": "sigmoied_threshold",
            "attention_block_rows":32,
            "attention_block_cols":32,
            "attention_lambda":1.0,
            "ampere_pruning_method": "disabled",
            "mask_init": "constant",
            "mask_scale": 0.0,
            "regularization": "l1",
            "regularization_final_lambda": 30,
            "attention_output_with_dense": 0
    }
    glue = GlueSparsePruning(param_dict)

    def hp_space(trial):
        return {}

    glue.run()

if __name__ == "__main__":
    main()


