import logging
import os
import json
from timeit import timeit
import torch
import numpy as np
import random
from datasets import load_metric, load_dataset
from pathlib import Path

from transformers import Trainer
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)
from nn_pruning.sparse_trainer import SparseTrainer, TimingModule
from nn_pruning.inference_model_patcher import optimize_model

# Preparation

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
## Define GLUE trainer

class GlueTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        #self.model_args = kwargs.pop("model_args")
        self.data_args = kwargs.pop("data_args")
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
    
    def evaluate(self, eval_dataset = None, eval_example=None, ignore_keys=None):
        data_args = self.data_args
        eval_dataset = self.additional_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]

        logger.info("*** Evaluate ***")

        ### MNLI double evaluation
        tasks = [data_args.dataset_name]
        eval_datasets = [eval_dataset]
        if data_args.dataset_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(self.additional_datasets["validation_mismatched"])
        
        eval_dataloaders = []
        for eval_dataset in eval_datasets:
            eval_dataloaders.append(self.get_eval_dataloader(eval_dataset))

        # Temporarily disable metric computation
        checkpoint_dir = self.checkpoint_dir()

        output0 = None
        for eval_dataloader, task in zip(eval_dataloaders, tasks):
            self.start_timer()

            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
            if output0 is None:
                output0 = output
            self.end_timer(len(eval_dataset),task)

            eval_results = output.metrics

            log_metrics = {f"eval_{task}_{k}": v for k,v in output.metrics.items()}
            self.log(log_metrics)

            output_eval_file = os.path.join(checkpoint_dir, f"eval_results_{task}.json")
            if self.is_world_process_zero():
                logger.info("***** Eval results {task} *****")
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
            # Removing the "label" columns because it contains -1 and Trainer won't like that.
            if "label" in test_dataset.column_names:
                test_dataset.remove_columns_("label")
            predictions = self.predict(test_dataset=test_dataset).predictions
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            predictions = np.squeeze(predictions, axis=1) if self.is_regression else np.argmax(predictions, axis=1)

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
        if not checkpoint_dir.exist():
            checkpoint_dir.mkdir()
        
        return checkpoint_dir
    
    def instrument_model(self,model):
        if self.args.optimize_model_before_eval != "disabled":
            model = optimize_model(self.model,self.args.optimize_model_before_eval)
        
        return TimingModule(model)
    

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

## Define Pruning Trainer

class GluePruningTrainer(SparseTrainer, GlueTrainer):
    def __init__(self, sparse_args, *args, **kwargs):
        GlueTrainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        We override the default loss in SparseTrainer because it throws an error when run without distillation
        """

        outputs = model(**inputs)
        labels = inputs["labels"]
        logits = outputs["logits"]
        logits = torch.argmax(logits, axis=-1)
        acc = (logits[:] == labels[:]).sum(axis=1, keepdims=True)
        correct_labels = acc.sum() / (labels.shape[0] * labels.shape[1])
        acc = (acc == labels.shape[1]).sum() / labels.shape[0]


        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        regu_loss, lamb, info = self.patch_coordinator.regularization_loss(model)
        for kind, values in info.items():
            if kind == "total":
                suffix = ""
            else:
                suffix = "_" + kind

            for k, v in values.items():
                self.metrics[k + suffix] += float(v)

        self.metrics["accuracy"] += acc
        self.metrics["correct_labels"] += correct_labels
        # self.metrics["ce_loss"] += float(loss.mean())
        # self.metrics["uniqueness"] += uniqueness
        self.loss_counter += 1

        loss = loss + regu_loss * lamb

        return (loss, outputs) if return_outputs else loss

## Define DataClass

class GlueDataset:

    def __init__(self, data_args, tokenizer):
        self.data_args = data_args
        self.tokenizer = tokenizer
    
    def create_datasets(self):
        data_args = self.data_args
        if data_args.dataset_name is not None:
            datasets = load_dataset("glue", data_args.dataset_name)
        elif data_args.train_file.endswith(".json"):
            raise RuntimeError("loading from json not available at the moment")
        else: raise RuntimeError("Please try loading from data_args")

        if data_args.dataset_name is not None:
            is_regression = data_args.dataset_name == "stsb"
            if not is_regression:
                label_list = datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                label_list = None
                num_labels = 1
        else:
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
        self.label_list = label_list
        self.num_labels = num_labels

    
        if data_args.dataset_name is not None:
            sentence1_key, sentence2_key = task_to_keys[data_args.dataset_name]
        else:
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

        if data_args.pad_to_max_length:
            padding = "max_length"
            max_length = data_args.max_seq_length
        else: #pad later, dynamically at batch creation
            padding = False
            max_length = None

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

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
        datasets = datasets
        train_dataset = datasets["train"]
        eval_dataset = datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"] 
        if data_args.dataset_name is not None:
            test_dataset = datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
        # Log a few random samples from training set: 
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        return train_dataset, eval_dataset, test_dataset
