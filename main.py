from numpy.core.fromnumeric import size
import torch
import datasets
import transformers
import os
import argparse
from transformers import Trainer, TrainerCallback
from nn_pruning.sparse_trainer import SparseTrainer
from nn_pruning.patch_coordinator import SparseTrainingArguments
from datasets import load_dataset
from data import get_dataset
from transformers import TrainingArguments
import torch 
# from transformers import AutoModelForCausalLM, AutoConfig
# from transformers import AutoConfig
from nn_pruning.patch_coordinator import ModelPatchingCoordinator
from nn_pruning.inference_model_patcher import optimize_model
from model import GPTNeoForCausalLM
import numpy as np
import copy
from torch import nn
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch GPT-Neo ft script')


parser.add_argument('--dataset_path', default=None, help='location of data corpus')
parser.add_argument('--tokenizer_path', required=True,  help='location of tokenizer')
parser.add_argument('--model_path', required=True, help='location of model')
parser.add_argument('--output_dir', default=None, help='location of output dir')
parser.add_argument('--save_model', action='store_true', help='save the net')

parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='epochs')

# parser.add_argument('--prune', action='store_true', help='simple prune test')
parser.add_argument('--dense_pruning_method', default="disabled", help='dense pruning method', choices=('disabled', 'topK', 'topK:1d_alt', 'threshold', 'sigmoied_threshold:1d_alt'))
parser.add_argument('--attention_pruning_method', default="disabled", help='attention pruning method', choices=('disabled', 'topK', 'threshold', 'sigmoied_threshold'))
parser.add_argument('--regularization', default="disabled", help='regularization method', choices=('disabled', 'l0', 'l1', "uniqueness"))
parser.add_argument('--train', action='store_true', help='train the net')
parser.add_argument('--evaluate', action='store_true', help='evaluate the net')

parser.add_argument('--train_samples', default=None, type=int, help='number of training samples to use')
parser.add_argument('--valid_samples', default=None, type=int, help='number of validation samples to use')


if __name__ == "__main__": 
    args = parser.parse_args()

    do_prune = args.dense_pruning_method != "disabled" or args.attention_pruning_method  != "disabled"

    datasets.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    print(f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__} and torch v{torch.__version__}")

    gptneo_name = args.model_path
    # gptneo_name = "EleutherAI/gpt-neo-125M"
    # gptneo_name = "EleutherAI/gpt-neo-2.7B"

    wikisql_train = get_dataset(args.tokenizer_path, "", "train", args.train_samples, 512, 512, False)
    wikisql_validation = get_dataset(args.tokenizer_path, "", "validation", args.valid_samples, 512, 512, False)
    wikisql_test = get_dataset(args.tokenizer_path, "", "test", args.valid_samples, 512, 512, False)

    log_df = []

    class LogDfCallback(TrainerCallback):
        """
        A bare :class:`~transformers.TrainerCallback` that just prints the logs.
        """

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            # _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                # print(logs)
                log_df.append(metrics)
            
    class PruningTrainer(SparseTrainer, Trainer):
        def __init__(self, sparse_args, *args, **kwargs):
            Trainer.__init__(self, *args, **kwargs)
            SparseTrainer.__init__(self, sparse_args)
            
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            We override the default loss in SparseTrainer because it throws an 
            error when run without distillation
            """
            outputs = model(**inputs)

            labels = inputs["labels"]
            logits = outputs["logits"]
            logits = torch.argmax(logits, axis=-1)
            acc = (logits[:] == labels[:]).sum(axis=1, keepdims=True)
            correct_labels = acc.sum() / (labels.shape[0] * labels.shape[1])
            acc = (acc == labels.shape[1]).sum() / labels.shape[0]

            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            uniqueness = outputs["uniqueness"].mean()
            regu_loss, lamb, info = self.patch_coordinator.regularization_loss(model)
            for kind, values in info.items():
                if kind == "total":
                    suffix = ""
                else:
                    suffix = "_" + kind

                for k, v in values.items():
                    self.metrics[k + suffix] += float(v)
            # self.metrics["ce_loss"] += float(loss.mean())
            self.metrics["accuracy"] += acc
            self.metrics["correct_labels"] += correct_labels
            self.metrics["uniqueness"] += uniqueness
            self.loss_counter += 1

            # loss = loss + regu_loss * lamb 
            loss = loss + regu_loss * lamb + uniqueness * lamb
            # print(loss)
            return (loss, outputs) if return_outputs else loss

        
        def _save(self, output_dir = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Saving model checkpoint to {output_dir}")
            self.model.save_pretrained(output_dir, state_dict=state_dict)
            if do_prune:
                print("Compiling model")
                model_copy = copy.deepcopy(self.model)
                self.patch_coordinator.compile_model(model_copy)
                compiled_dir = os.path.join(output_dir, "compiled")
                print(f"Saving compiled model checkpoint to {compiled_dir}")
                model_copy.save_pretrained(compiled_dir, state_dict=state_dict)

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))



    sparse_args = SparseTrainingArguments()

    initial_threshold = 1.0
    # final_threshold = 0.5 # top 50% of topk
    final_threshold = 0.1 # top 10% of topk
    if args.dense_pruning_method == "threshold":
        initial_threshold = 0.001
        final_threshold = .001 # this is for uniqueness
    elif "sigmoied_threshold" in args.dense_pruning_method:
        initial_threshold = 0
        final_threshold = 0.1 # over .1 for sigmoid

    regularization_final_lambda = 0
    if args.regularization != "disabled":
        # regularization_final_lambda = 10
        regularization_final_lambda = 2

    hyperparams = {
        "dense_pruning_method": args.dense_pruning_method, 
        "attention_pruning_method": args.attention_pruning_method, 
        "regularization": args.regularization,
        "regularization_final_lambda": regularization_final_lambda,
        "ampere_pruning_method": "disabled",
        "initial_threshold": initial_threshold, 
        "final_threshold": final_threshold, 
        "initial_warmup": 1,
        "final_warmup": 3,
        "attention_block_rows":32,
        "attention_block_cols":32,
        "attention_output_with_dense": 0,
        "save_uniqueness": args.regularization == "uniqueness",
        
    }

    for k,v in hyperparams.items():
        if hasattr(sparse_args, k):
            setattr(sparse_args, k, v)
        else:
            print(f"sparse_args does not have argument {k}")


    learning_rate = 2e-4
    # learning_rate = 2e-6
    n_gpu = torch.cuda.device_count()
    batch_size = args.batch_size
    epoch_steps = len(wikisql_train) // (batch_size*n_gpu)
     
    num_train_epochs = args.epochs 
    logging_steps = epoch_steps
    # warmup for 10% of training steps
    warmup_steps = logging_steps * num_train_epochs * 0.1  # 10 %
    # eval_steps = int(epoch_steps * num_train_epochs / 12)   # eval 12 times
    eval_steps = int(epoch_steps*5)   # eval every 5 epochs


    print("eval steps", eval_steps)
    print("batch_size", batch_size)
    print("epoch_steps", epoch_steps)
    print("n_gpu", n_gpu)

    save_strategy = "no"
    if args.save_model:
        save_strategy = "steps"
    if args.output_dir is None:
        output_dir = "checkpoints"
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = os.path.join(args.output_dir, "checkpoints")

    training_args = TrainingArguments(
        output_dir=output_dir,
        # output_dir=None,
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps= eval_steps,
        save_strategy=save_strategy,
        save_steps = eval_steps,
        # gradient_accumulation_steps=1,
        # eval_accumulation_steps=10,
        eval_accumulation_steps=2,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.1,
        warmup_steps=warmup_steps,
        # weight_decay=1e-4,
        logging_steps=logging_steps,
        # disable_tqdm=True,
        disable_tqdm=False,
        report_to=None,
        # adam_beta1=.9,
        # adam_beta2=.999,
    )





    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mpc = ModelPatchingCoordinator(
        sparse_args=sparse_args, 
        device=device, 
        cache_dir="checkpoints", 
        model_name_or_path=gptneo_name,
        logit_names="logits", 
        teacher_constructor=None)

    gptneo_model = GPTNeoForCausalLM.from_pretrained(gptneo_name).to(device)




    if args.train:
        with torch.no_grad():
            # gptneo_model.transformer.wte.weight.data.normal_(mean=0.0, std=0.02)

            embed_shape = gptneo_model.transformer.wte.weight.shape
            decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
            decoder.weight = gptneo_model.transformer.wte.weight  # Tied weights with input
            gptneo_model.set_output_embeddings(decoder)

        mpc.patch_model(gptneo_model)

    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=-1)
        # predictions, labels = predictions[..., :, :-1], labels[..., :, 1:]
        # acc = (predictions == labels).sum(axis=1, keepdims=True) == labels.shape[1]
        # acc = (predictions == labels).sum(axis=1, keepdims=True)
        # print(acc)

        real = wikisql_validation.tokenizer.decode(labels[0])
        pred = wikisql_validation.tokenizer.decode(predictions[0])
        ridx = real.find("<|endoftext|>")
        pidx = real.find("<|endoftext|>")
        print()
        print("SAMPLE", real[:ridx])
        print("PREDICTION", pred[:pidx])
        print()
        # print("sample", real, "pred", pred)


        acc = (predictions[:] == labels[:]).sum(axis=1, keepdims=True) == labels.shape[1]
        return {"accuracy": acc.sum() / labels.shape[0]}


    trainer = PruningTrainer(
        sparse_args=sparse_args,
        args=training_args,
        model=gptneo_model,
        train_dataset=wikisql_train,
        eval_dataset=wikisql_validation,
        callbacks=[LogDfCallback]
    )

    trainer.set_patch_coordinator(mpc)

    if args.train:

        print("training")
        trainer.train()

        print("evaluating")
        results = trainer.evaluate()
        print("results")
        print(results)

        if args.output_dir:
            print("saving results")
            log_file = os.path.join(args.output_dir, 'log.df')
            pd.DataFrame(log_df).to_pickle(log_file)

    if args.evaluate:
        trainer = PruningTrainer(
            sparse_args=sparse_args,
            args=training_args,
            model=gptneo_model,
            train_dataset=wikisql_train,
            eval_dataset=wikisql_validation,
        )
        trainer.set_patch_coordinator(mpc)
        print("evaluating validation set ")
        results = trainer.evaluate()
        print("results")
        print(results)
        
        trainer = PruningTrainer(
            sparse_args=sparse_args,
            args=training_args,
            model=gptneo_model,
            train_dataset=wikisql_train,
            eval_dataset=wikisql_test,
        )
        trainer.set_patch_coordinator(mpc)

        print("evaluating test set ")
        results = trainer.evaluate()
        print("results")
        print(results)
    


    if do_prune:
        print("evaluating pruning")

        print("compiling")
        mpc.compile_model(trainer.model)

        print("optimizing model")

        pruned_gptneo_model = optimize_model(trainer.model, "dense")

        size_diff = pruned_gptneo_model.num_parameters() / gptneo_model.num_parameters()

        print(f"reduced model to {size_diff} of original size")
        
        trainer = PruningTrainer(
            sparse_args=sparse_args,
            args=training_args,
            model=pruned_gptneo_model,
            train_dataset=wikisql_train,
            eval_dataset=wikisql_validation,
        )

        trainer.set_patch_coordinator(mpc)

        print("pruned evaluation")

        pruned_results = trainer.evaluate()
        print(pruned_results)

        print("done")

        
    print("done")