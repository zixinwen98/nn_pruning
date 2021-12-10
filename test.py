from numpy.core.fromnumeric import size
import torch
import datasets
import transformers
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch GPT-Neo ft script')


parser.add_argument('--dataset_path', default=None, help='location of data corpus')
parser.add_argument('--tokenizer_path', required=True,  help='location of tokenizer')
parser.add_argument('--model_path', required=True, help='location of model')
parser.add_argument('--save_path', default=None, help='location of save model')

parser.add_argument('--batch_size', required=True, type=int, help='batch size')
parser.add_argument('--epochs', default=20, type=int, help='epochs')

parser.add_argument('--prune', action='store_true', help='simple prune test')
parser.add_argument('--train', action='store_true', help='train the net')
parser.add_argument('--evaluate', action='store_true', help='evaluate the net')

parser.add_argument('--train_samples', default=None, type=int, help='number of training samples to use')
parser.add_argument('--valid_samples', default=None, type=int, help='number of validation samples to use')


if __name__ == "__main__": 
    args = parser.parse_args()




    datasets.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    print(f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__} and torch v{torch.__version__}")

    from datasets import load_dataset

    # wikisql = load_dataset("wikisql")
    # print(len(wikisql["train"]))
    # 0/0



    # from transformers import AutoTokenizer

    gptneo_name = args.model_path
    # gptneo_name = "EleutherAI/gpt-neo-125M"
    # gptneo_name = "EleutherAI/gpt-neo-2.7B"

    # tokenizer = AutoTokenizer.from_pretrained(gptneo_name)

    # def tokenize_and_encode(examples): 
    #     return tokenizer(examples['question'], examples['sql']['human_readable'], max_length=512, padding="max_length")

    # wikisql_enc = wikisql.map(tokenize_and_encode, batched=True)

    from data import get_dataset

    wikisql_train = get_dataset(args.tokenizer_path, "", "train", args.train_samples, 512, 512, False)
    wikisql_validation = get_dataset(args.tokenizer_path, "", "validation", args.valid_samples, 512, 512, False)
    wikisql_test = get_dataset(args.tokenizer_path, "", "test", args.valid_samples, 512, 512, False)

    # wikisql_train = get_dataset("train", 20, 512, 512, False)
    # wikisql_validation = get_dataset("train", 20, 512, 512, False)
    # wikisql_validation = get_dataset("validation", None, 512, 512, False)
    # wikisql_validation = get_dataset("validation", 200, 512, 512, False)
    # wikisql_validation = get_dataset("validation", 20, 512, 512, False)




    from transformers import Trainer
    from nn_pruning.sparse_trainer import SparseTrainer

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

            # print(inputs.keys())
            # print(outputs.keys())
            # 0/0
            labels = inputs["labels"]
            logits = outputs["logits"]
            logits = torch.argmax(logits, axis=-1)
            # acc = (logits[:] == labels[:]).sum(axis=1, keepdims=True) == labels.shape[1]
            acc = (logits[:] == labels[:]).sum(axis=1, keepdims=True)
            correct_labels = acc.sum() / (labels.shape[0] * labels.shape[1])
            acc = (acc == labels.shape[1]).sum() / labels.shape[0]

            # return {"accuracy": acc.sum() / labels.shape[0]}

            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # self.metrics["ce_loss"] += float(loss.mean())
            self.metrics["accuracy"] += acc
            self.metrics["correct_labels"] += correct_labels
            self.loss_counter += 1
            # print(loss)
            return (loss, outputs) if return_outputs else loss


    from nn_pruning.patch_coordinator import SparseTrainingArguments

    sparse_args = SparseTrainingArguments()

    dense_pruning_method = "disabled"
    attention_pruning_method = "disabled"
    if args.prune:
        dense_pruning_method = "topK:1d_alt"
        attention_pruning_method = "topK"

    hyperparams = {
        "dense_pruning_method": dense_pruning_method, 
        "attention_pruning_method": attention_pruning_method, 
        # "dense_pruning_method": "disabled", 
        # "attention_pruning_method": "disabled", 
        "ampere_pruning_method": "disabled",
        "initial_threshold": 1.0, 
        "final_threshold": 0.5, 
        "initial_warmup": 1,
        "final_warmup": 3,
        "attention_block_rows":32,
        "attention_block_cols":32,
        "attention_output_with_dense": 0
    }

    for k,v in hyperparams.items():
        if hasattr(sparse_args, k):
            setattr(sparse_args, k, v)
        else:
            print(f"sparse_args does not have argument {k}")

    from transformers import TrainingArguments

    learning_rate = 2e-4
    n_gpu = torch.cuda.device_count()
    batch_size = args.batch_size
    epoch_steps = len(wikisql_train) // (batch_size*n_gpu)
     
    # batch_size = 16
    # batch_size = 1
    # learning_rate = 2e-6
    # learning_rate = 2e-3
    num_train_epochs = args.epochs 
    logging_steps = epoch_steps
    eval_steps = int(epoch_steps * num_train_epochs / 12)   # eval 12 times
    # eval_steps = int(epoch_steps*5)   # eval every 5 epochs
    print("eval steps", eval_steps)
    print("batch_size", batch_size)
    print("epoch_steps", epoch_steps)
    print("n_gpu", n_gpu)
    # warmup for 10% of training steps
    warmup_steps = logging_steps * num_train_epochs * 0.1  # 10 %

    save_strategy = "steps"
    if args.save_path is None:
        save_strategy = "no"
        output_dir = "checkpoints"
    else:
        output_dir = os.path.join(args.save_path, "checkpoints")

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





    import torch 
    # from transformers import AutoModelForCausalLM, AutoConfig
    from transformers import AutoConfig
    from nn_pruning.patch_coordinator import ModelPatchingCoordinator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mpc = ModelPatchingCoordinator(
        sparse_args=sparse_args, 
        device=device, 
        cache_dir="checkpoints", 
        model_name_or_path=gptneo_name,
        logit_names="logits", 
        teacher_constructor=None)

    # gptneo_model = AutoModelForCausalLM.from_pretrained(gptneo_name).to(device)
    # config = AutoConfig.from_pretrained(gptneo_name)
    # gptneo_model = GPTNeoForCausalLM(config)
    from model import GPTNeoForCausalLM
    gptneo_model = GPTNeoForCausalLM.from_pretrained(gptneo_name).to(device)


    from torch import nn

    if args.train:
        with torch.no_grad():
            # gptneo_model.transformer.wte.weight.data.normal_(mean=0.0, std=0.02)

            embed_shape = gptneo_model.transformer.wte.weight.shape
            decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
            decoder.weight = gptneo_model.transformer.wte.weight  # Tied weights with input
            gptneo_model.set_output_embeddings(decoder)

        mpc.patch_model(gptneo_model)



    import numpy as np
    from datasets import load_metric

    accuracy_score = load_metric('accuracy')

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
    )

    trainer.set_patch_coordinator(mpc)

    if args.train:
        print("training")
        trainer.train()

        print("evaluating")
        results = trainer.evaluate()
        print("results")
        print(results)

    if args.evaluate:
        # from nn_pruning.inference_model_patcher import optimize_model

        # pruned_gptneo_model = optimize_model(trainer.model, "dense")
        # mpc.compile_model(gptneo_model)
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
    


    if args.prune:
        print("evaluating pruning")

        print("compiling")
        mpc.compile_model(trainer.model)

        print("optimizing model")
        from nn_pruning.inference_model_patcher import optimize_model

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