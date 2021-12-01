import torch
import datasets
import transformers
datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
print(f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__} and torch v{torch.__version__}")

from datasets import load_dataset

# wikisql = load_dataset("wikisql")
# print(len(wikisql["train"]))
# 0/0



# from transformers import AutoTokenizer

gptneo_name = "EleutherAI/gpt-neo-125M"
# tokenizer = AutoTokenizer.from_pretrained(gptneo_name)

# def tokenize_and_encode(examples): 
#     return tokenizer(examples['question'], examples['sql']['human_readable'], max_length=512, padding="max_length")

# wikisql_enc = wikisql.map(tokenize_and_encode, batched=True)

from data import get_dataset

wikisql_train = get_dataset("train", None, 512, 512, False)
wikisql_validation = get_dataset("validation", None, 512, 512, False)




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

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        self.metrics["ce_loss"] += float(loss)
        self.loss_counter += 1
        return (loss, outputs) if return_outputs else loss


from nn_pruning.patch_coordinator import SparseTrainingArguments

sparse_args = SparseTrainingArguments()
sparse_args

hyperparams = {
    "dense_pruning_method": "topK:1d_alt", 
    "attention_pruning_method": "topK", 
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

batch_size = 16
learning_rate = 2e-5
num_train_epochs = 6
logging_steps = len(wikisql_train) // batch_size
# warmup for 10% of training steps
warmup_steps = logging_steps * num_train_epochs * 0.1

args = TrainingArguments(
    output_dir="checkpoints",
    evaluation_strategy="epoch",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=0.01,
    logging_steps=logging_steps,
    disable_tqdm=False,
    report_to=None
)





import torch 
from transformers import AutoModelForCausalLM
from nn_pruning.patch_coordinator import ModelPatchingCoordinator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mpc = ModelPatchingCoordinator(
    sparse_args=sparse_args, 
    device=device, 
    cache_dir="checkpoints", 
    model_name_or_path=gptneo_name,
    logit_names="logits", 
    teacher_constructor=None)

gptneo_model = AutoModelForCausalLM.from_pretrained(gptneo_name).to(device)
mpc.patch_model(gptneo_model)



import numpy as np
from datasets import load_metric

accuracy_score = load_metric('accuracy')

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)


trainer = PruningTrainer(
    sparse_args=sparse_args,
    args=args,
    model=gptneo_model,
    train_dataset=wikisql_train,
    eval_dataset=wikisql_validation,
    # tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # load_best_model_at_end=True,
    # metric_for_best_model="accuracy"
)

trainer.set_patch_coordinator(mpc)

trainer.train();





0/0

boolq = load_dataset("super_glue", "boolq")

boolq.rename_column_("label", "labels")

from transformers import AutoTokenizer

bert_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)

def tokenize_and_encode(examples): 
    return tokenizer(examples['question'], examples['passage'], truncation="only_second")

boolq_enc = boolq.map(tokenize_and_encode, batched=True)

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

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        self.metrics["ce_loss"] += float(loss)
        self.loss_counter += 1
        return (loss, outputs) if return_outputs else loss


from nn_pruning.patch_coordinator import SparseTrainingArguments

sparse_args = SparseTrainingArguments()
sparse_args

hyperparams = {
    "dense_pruning_method": "topK:1d_alt", 
    "attention_pruning_method": "topK", 
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

batch_size = 16
learning_rate = 2e-5
num_train_epochs = 6
logging_steps = len(boolq_enc["train"]) // batch_size
# warmup for 10% of training steps
warmup_steps = logging_steps * num_train_epochs * 0.1

args = TrainingArguments(
    output_dir="checkpoints",
    evaluation_strategy="epoch",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=0.01,
    logging_steps=logging_steps,
    disable_tqdm=False,
    report_to=None
)

import torch 
from transformers import AutoModelForSequenceClassification
from nn_pruning.patch_coordinator import ModelPatchingCoordinator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mpc = ModelPatchingCoordinator(
    sparse_args=sparse_args, 
    device=device, 
    cache_dir="checkpoints", 
    model_name_or_path=bert_ckpt,
    logit_names="logits", 
    teacher_constructor=None)

bert_model = AutoModelForSequenceClassification.from_pretrained(bert_ckpt).to(device)
mpc.patch_model(bert_model)

import numpy as np
from datasets import load_metric

accuracy_score = load_metric('accuracy')

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)


trainer = PruningTrainer(
    sparse_args=sparse_args,
    args=args,
    model=bert_model,
    train_dataset=boolq_enc["train"],
    eval_dataset=boolq_enc["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # load_best_model_at_end=True,
    # metric_for_best_model="accuracy"
)

trainer.set_patch_coordinator(mpc)

trainer.train();