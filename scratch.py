from nn_pruning.inference_model_patcher import optimize_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.gpt_neo.modeling_gpt_neo import *
from transformers import pipeline
from datasets import load_dataset
import torch

# quick test

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# tokenizer.pad_token = "[PAD]"
# test = tokenizer("these are words ", " and more words", max_length=20, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True)
# test = tokenizer.decode(test["input_ids"][0])
# print(test)
# 0/0
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_config(config)

# model.save_pretrained("/home/misantac/teamdrive/inflimitmsr/gpt_neo/")
# tokenizer.save_pretrained("/home/misantac/teamdrive/inflimitmsr/gpt_neo/")
# from datasets import load_dataset
# wikisql =  load_dataset('wikisql')
# wikisql.save_to_disk("/home/misantac/teamdrive/inflimitmsr/gpt_neo/wikisql")
# 0/0

# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(params)
# 0/0

# test = optimize_model(model, "dense", clone=False)

for name, module in model.named_modules():
    print(name)
# print([module for module in model.modules()])
0/0

prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
          "researchers was the fact that the unicorns spoke perfect English."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

test = model.forward(input_ids)

print(test.shape)
0/0

# gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
# gen_text = tokenizer.batch_decode(gen_tokens)[0]
# print(gen_text)

# print([module for module in model.modules()])