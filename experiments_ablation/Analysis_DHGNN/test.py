import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import re
import random
import numpy as np
import json
import shutil
from functools import partial
import torch
import datasets
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import argparse
import deepspeed
from PIL import Image
# from src.utils import *
# from src.CustomQwen2ForCausalLM import *
from transformers import Qwen2TokenizerFast, Qwen2Config, Qwen2ForCausalLM, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
torch.cuda.empty_cache()



parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', default="", type=str, help="")
parser.add_argument('--datasets_name', default="", type=str, help="")
parser.add_argument('--output_dir', default="", type=str, help="")
parser.add_argument('--model_path', default="", type=str, help="")
parser.add_argument('--model_name', default="", type=str, help="")
parser.add_argument('--system_prompt', default="", type=str, help="")
parser.add_argument('--template_prompt', default="", type=str, help="")
parser.add_argument('--is_training', default=0, type=int, help="")
parser.add_argument('--max_time_len', default=0, type=int, help="")
parser.add_argument('--batch_size', default=0, type=int, help="")
parser.add_argument('--device_id', default=-1, type=int, help="")
parser.add_argument('--max_new_tokens', default="", type=int, help="")
parser.add_argument('--do_sample', default=0, type=int, help="")
parser.add_argument('--flag', default="", type=str, help="")
parser.add_argument('--cases', default=None, type=str, help="")
args = parser.parse_args()
args.output_dir = os.path.join(args.output_dir, args.flag)
args.logging_dir = os.path.join(args.output_dir, "logging.json")
print(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
    json.dump(vars(args), f)

if args.cases == "NODHGNN":
    from src_NODHGNN.utils import *
    from src_NODHGNN.CustomQwen2ForCausalLM import *
elif args.cases == "NODifferential":
    from src_NODifferential.utils import *
    from src_NODifferential.CustomQwen2ForCausalLM import * 
elif args.cases == "SingleS3" or args.cases == "SingleS7":
    from src.utils import *
    from src.CustomQwen2ForCausalLM import *  
else:
    raise ValueError(f"cases {args.cases} is not defined!!!")
    

# data load
with open(os.path.join(args.datasets_path, args.datasets_name), "r", encoding="utf-8") as file:
    dataset = json.load(file)
    for item in dataset:
        item["family"] = "<"+item["family"]+">"
dataset = Dataset.from_list(dataset)
print(dataset)


# Initialize tokenizer and model
tokenizer = Qwen2TokenizerFast.from_pretrained(pretrained_model_name_or_path=args.model_path)
if args.device_id==-1:
    model = CustomQwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(args.model_path, args.model_name), torch_dtype="auto", device_map="auto").to(torch.bfloat16)
else:
    model = CustomQwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(args.model_path, args.model_name), torch_dtype="auto").to(torch.bfloat16).cuda(args.device_id)
print(model)



model.eval()
with torch.no_grad():
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CustomDataCollator(tokenizer, args.system_prompt, args.template_prompt, args.max_time_len, args.is_training, 0, 0, 0, model.config.family_tokens, args.datasets_path, args.datasets_name, model.config),
    )
    
    ground_predict = {}; ground_predict["ground"]=[]; ground_predict["predict"]=[]; ground_predict["acc"]=0
    for i, (batch_data, features) in enumerate(dataloader):
        for key in batch_data.keys():
            if isinstance(batch_data[key], list):
                batch_data[key] = [tensor.to(model.device) for tensor in batch_data[key]]
            else:
                batch_data[key] = batch_data[key].to(model.device)
        inputs = {key: batch_data[key] for key in batch_data.keys()}
        do_sample = True if args.do_sample==1 else False
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=do_sample)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"Finished: {i}\n")
        ground = [item["family"] for item in features]
        predict = [item for item in output_text]
        print(f"ground_true: {ground}\n")
        print(f"predict: {predict}\n")
        ground_predict["ground"] += ground
        ground_predict["predict"] += predict
    acc = sum(1 for x, y in zip(ground_predict["ground"], ground_predict["predict"]) if x == y) / len(ground_predict["ground"])
    ground_predict["acc"] = acc
with open(os.path.join(args.output_dir, "results_file.txt"), 'w', encoding='utf-8') as json_file:
    json.dump(ground_predict, json_file, ensure_ascii=False, indent=4)
