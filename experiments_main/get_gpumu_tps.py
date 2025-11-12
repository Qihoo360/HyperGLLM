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
from src.utils import *
from src.CustomQwen2ForCausalLM import *
from transformers import Qwen2TokenizerFast, Qwen2Config, Qwen2ForCausalLM, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
torch.cuda.empty_cache()






# data load
datasets_path = "../datasets/"
datasets_name = "TEST.json"
with open(os.path.join(datasets_path, datasets_name), "r", encoding="utf-8") as file:
    dataset = json.load(file)
    for item in dataset:
        item["family"] = "<"+item["family"]+">"
dataset = Dataset.from_list(dataset)
print(dataset)

# tokenizer、model
model_path = "./trainer_dir_paramS48HY4/"
model_name = "checkpoint-30000"
tokenizer = Qwen2TokenizerFast.from_pretrained(pretrained_model_name_or_path=model_path)
print(tokenizer)
model = CustomQwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(model_path, model_name), torch_dtype="auto", device_map="auto").to(torch.bfloat16)
print(model)

system_prompt = ""
template_prompt = "From a computer security perspective, what are the behavior types of the following logs collected by an Endpoint Detection and Response (EDR) system?\n {input}"
max_time_len = 2048
is_training = 0
dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=CustomDataCollator(tokenizer, system_prompt, template_prompt, max_time_len, is_training, 0, 0, 0, model.config.family_tokens, datasets_path, datasets_name, model.config),
)



#We use 1024 event points to approximate 1024K tokens (typically, the number of tokens per event exceeds 1K). "runs" refers to the number of executions, and we ultimately take the average.
lens = 1024; runs=100 
output_texts = []; times = []
model.eval()
with torch.no_grad():
    for i, (batch_data, _) in enumerate(dataloader):
        for key in batch_data.keys():
            if isinstance(batch_data[key], list):
                batch_data[key] = [tensor.to(model.device) for tensor in batch_data[key]]
            else:
                batch_data[key] = batch_data[key].to(model.device)
        inputs = {key: batch_data[key] for key in batch_data.keys()}
        indices = np.linspace(0, inputs["list_all_values"][0].shape[0] - 1, lens, dtype=int)
        inputs["list_all_values"] = [inputs["list_all_values"][k][indices] for k in range(len(inputs["list_all_values"]))]
        inputs["list_all_mask"] = [inputs["list_all_mask"][k][indices] for k in range(len(inputs["list_all_mask"]))]
        inputs["segs_num"] = inputs["segs_num"].fill_(lens)
        

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_texts += output_text

        end_event.record()
        torch.cuda.synchronize()
        generate_time_ms = start_event.elapsed_time(end_event) # time（ms）
        times.append(generate_time_ms)
        
        output_texts_ = []; times_ = []
        if i==runs-1:
            for idx, (item, time) in enumerate(zip(output_texts,times)):
                if idx<=4:
                    continue
                if item.find("<")==-1 and item.find(">")==-1:
                    continue
                output_texts_.append(item)
                times_.append(time)
            print(output_texts_)
            print(times_)
            print(sum(times_)/len(times_))
            break