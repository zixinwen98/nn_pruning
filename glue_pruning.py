## Import Modules
import torch
import numpy as np
import pandas as pd
import argparse
from datasets import load_metric
from transformers import (
    RobertaTokenizer,
    default_data_collator,
    EvalPrediction,
    AutoConfig,
    TrainerCallback,
    AutoModelForSequenceClassification,
)

import logging
import os

from nn_pruning.inference_model_patcher import optimize_model
from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments
from glue_utils import GlueDataset, GluePruningTrainer
from model_roberta import RobertaForSequenceClassification
from training_args import GlueDataTrainingArguments, PruningTrainingArguments

## Preparation

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

os.makedirs('checkpoints', exist_ok=True)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Args():
    parser = argparse.ArgumentParser(description='PyTorch Roberta for GLUE+Pruning task')
    parser.add_argument('--task_name', default='mnli', help='GLUE task choice', choices=('cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'))
    parser.add_argument('--dataset_path', default=None, help='location of data corpus')
    parser.add_argument('--model_name_or_path', default='roberta-base', help='name/location of model')
    parser.add_argument('--output_dir', default=None, help='location of output dir')
    parser.add_argument('--save_model', action='store_true', help='save the net')
    parser.add_argument('--seed', default=0, type=int, help='Random Seed')

    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--regu_lambda', default=.1, type=float, help='regularization lambda')
    parser.add_argument('--label_smoothing', default=0.2, type=float, help='label smoothing')
    parser.add_argument('--prune_leftover', default=.5, type=float, help='amount of params left over after pruning')
    
    # parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--schedule', default="linear", help='schedule type', choices=('linear', 'cos', 'constant'))
    parser.add_argument('--token_max_len', default=512, type=int, help='token max len')

    # parser.add_argument('--prune', action='store_true', help='simple prune test')
    parser.add_argument('--dense_pruning_method', default="disabled", help='dense pruning method', choices=('disabled', 'uniqueness', 'topK', 'magnitude', 'threshold', 'sigmoied_threshold', 'random', "l0"))
    parser.add_argument('--dense_pruning_submethod', default="default", help='dense pruning submethod', choices=('default', '1d', '1d_alt'))
    parser.add_argument('--attention_pruning_method', default="disabled", help='attention pruning method', choices=('disabled', 'uniqueness', 'topK', 'magnitude', 'threshold', 'sigmoied_threshold'))
    parser.add_argument('--regularization', default="disabled", help='regularization method', choices=('disabled', 'l0', 'l1'))

    parser.add_argument('--do_train', action='store_true', help='train the net')
    parser.add_argument('--do_eval', action='store_true', help='evaluate the net')
    parser.add_argument('--do_prune', action='store_true', help='prune the net')
    parser.add_argument('--per_device_train_batch_size', default=16, type=int, help='train batch size per device')
    parser.add_argument('--per_device_eval_batch_size', default=16, type=int, help='eval batch size per device')

    parser.add_argument('--train_samples', default=None, type=int, help='number of training samples to use')
    parser.add_argument('--valid_samples', default=None, type=int, help='number of validation samples to use')

    parser.add_argument('--logging_steps', default=10, type=int, help='log every number of steps')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='epochs')
    parser.add_argument('--report_to', default=None, type=str, help='report to wandb')
    parser.add_argument('--max_seq_length', default=128, type=int, help='max sequence length in a batch')

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--warmup_steps", default=1200, type=int)

    return parser.parse_args()

log_df = []

class LogDfCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            # print(logs)
            log_df.append({**metrics, **vars(args)})

## Set up All the arguments

#train_args: TrainingArguments = None, 
#data_args: DataTrainingArguments = None, 
#sparse_args: SparseTrainingArguments = None,
#model_args: ModelArguments = None,

if __name__ == "__main__":

    args = Args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay                                                
    num_train_epochs = args.num_train_epochs
    logging_steps = args.logging_steps
    do_train = args.do_train
    do_eval = args.do_eval
    # warmup for 10% of training steps
    warmup_steps = logging_steps * num_train_epochs * 0.1

    if args.output_dir is None:
        output_dir = "glue_checkpoints"
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = os.path.join(args.output_dir, "glue_checkpoints")

    

    train_args = PruningTrainingArguments(
        do_train=do_train,
        do_eval=do_eval,
        output_dir=output_dir,
        evaluation_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        disable_tqdm=False,
        #report_to=args.report_to
    )

    sparse_args = SparseTrainingArguments()

    initial_threshold = 1.0
    # final_threshold = 0.1
    final_threshold = args.prune_leftover
    if "threshold" in args.dense_pruning_method:
        initial_threshold = 0
        final_threshold = .1 # different meaning for movemenet
    elif "sigmoied_threshold" in args.dense_pruning_method:
        initial_threshold = 0
        final_threshold = 0.1 # different meaning for movemenet
    regularization_final_lambda = 0
    if args.regularization != "disabled":
        regularization_final_lambda = args.regu_lambda


    hyperparams = {
        "dense_pruning_method": args.dense_pruning_method + ":" + args.dense_pruning_submethod, 
        "attention_pruning_method": args.attention_pruning_method, 
        "regularization": args.regularization,
        "regularization_final_lambda": regularization_final_lambda,
        "ampere_pruning_method": "disabled",
        "initial_threshold": initial_threshold, 
        "final_threshold": final_threshold, 
        "initial_warmup": 1,
        "final_warmup": 2,
        "warmup_steps": args.warmup_steps,
        "attention_block_rows":32,
        "attention_block_cols":32,
        # "attention_block_rows":1,
        # "attention_block_cols":1,
        "attention_output_with_dense": 0,
        "schedule_type": args.schedule,
        "linear_min_parameters": args.prune_leftover,
        "mask_init": "constant",
        "mask_scale": 0.0,
        "mask_scores_learning_rate": 0.01,
        "max_grad_norm": 1.0,
    }

    if "threshold" in args.dense_pruning_method or "sigmoied_threshold" in args.dense_pruning_method:
        #hyperparams["mask_scores_learning_rate"] = 10
        hyperparams["mask_scores_learning_rate"] = 0.01
    # else:
    #     hyperparams["mask_scores_learning_rate"] = 0

    for k,v in hyperparams.items():
        if hasattr(sparse_args, k):
            setattr(sparse_args, k, v)
        else:
            print(f"sparse_args does not have argument {k}")

    ## Load GLUE datasets
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    data_args = GlueDataTrainingArguments(
        dataset_name = args.task_name,
        max_seq_length=args.max_seq_length,
        pad_to_max_length=True
    )

    dataset = GlueDataset(
        data_args=data_args,
        tokenizer=tokenizer 
    )

    train_dataset, eval_dataset, test_dataset = dataset.create_datasets()

    ## Set up Model and Patch Coordinator

    model_name_or_path = args.model_name_or_path

    if model_name_or_path == "roberta-base":
        config_path = model_name_or_path

        model_config = AutoConfig.from_pretrained(
            config_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
        )

    if "roberta" in args.model_name_or_path:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=model_config).to(device)
    elif args.model_name_or_path is not None:
        model = AutoModelForSequenceClassification.from_prertained(args.model_name_or_path).to(device)

    mpc = ModelPatchingCoordinator(
        sparse_args=sparse_args, 
        device=device, 
        cache_dir="glue_checkpoints", 
        model_name_or_path=args.model_name_or_path,
        logit_names="logits", 
        teacher_constructor=None
    )

    mpc.patch_model(model)

    ## Set up Pruning Trainer and do pruning

    # Get the metric function
    if data_args.dataset_name is not None:
        metric = load_metric("glue", data_args.dataset_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if dataset.is_regression else np.argmax(preds, axis=1)
        if data_args.dataset_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif dataset.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}



    # Initialize our Trainer
    pruning_trainer = GluePruningTrainer(
        model=model,
        sparse_args=sparse_args,
        args=train_args,
        data_args=data_args,
        train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=eval_dataset if train_args.do_eval else None,
        additional_datasets=dataset.datasets,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    pruning_trainer.set_patch_coordinator(mpc)

    ## Training

    if args.do_train:
        print("Training Begin")
        pruning_trainer.train()

        print("Evaluating")
        results = pruning_trainer.evaluate()

        if args.output_dir:
            print("saving results")
            log_file = os.path.join(args.output_dir, 'log.df')
            pd.DataFrame(log_df).to_pickle(log_file)

    
    if args.do_eval:
        pruning_trainer = GluePruningTrainer(
            model=model,
            sparse_args=sparse_args,
            args=train_args,
            data_args=data_args,
            train_dataset=None,
            eval_dataset=test_dataset if test_dataset is not None else eval_dataset,
            additional_datasets=dataset.datasets,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=default_data_collator if data_args.pad_to_max_length else None,
        )

        pruning_trainer.set_patch_coordinator(mpc)
        results = pruning_trainer.evaluate()
        print('Results')
        print(results)

    ## Import optimize_model to see the pruning result
    if args.do_prune:
        print("evaluating pruning")

        print("compiling")
        mpc.compile_model(pruning_trainer.model)

        print("optimizing model")

        pruned_model = optimize_model(pruning_trainer.model, "dense")

        size_diff = pruned_model.num_parameters() / model.num_parameters()

        print(f"reduced model to {size_diff} of original size")

        test_trainer = GluePruningTrainer(
            model=pruned_model,
            sparse_args=sparse_args,
            args=train_args,
            data_args=data_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            additional_datasets=dataset.datasets,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=default_data_collator if data_args.pad_to_max_length else None,
        )

        test_trainer.set_patch_coordinator(mpc)

        print("pruned evaluation")

        pruned_results = test_trainer.evaluate()
        print(pruned_results)

        print("done")

        
    print("All done")
    