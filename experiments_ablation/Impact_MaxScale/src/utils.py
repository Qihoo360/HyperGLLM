import os
import re
import string
import jieba
import difflib
from difflib import SequenceMatcher
import numpy as np
from typing import List
from collections import Counter
from rouge import Rouge
from dataclasses import dataclass, field, asdict
import pandas as pd
import json
from math import ceil
from typing import Optional, List, Dict, Any, Union, Mapping, Iterable
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import random
import pathlib
import sys
import pytz
from datetime import datetime
from transformers import DefaultDataCollator, TrainerCallback, Trainer
from .data_augmentation import *
import pickle
import gzip






used_keys = {'action_type': [6, 12.09, 14.0, 14], 'addr_family': [7, 7.41, 8.0, 8],
    'child_command_line': [1, 85.68, 127.0, 20638], 'child_process_build_name': [1, 11.34, 21.0, 127],
    'child_process_cert_hash': [40, 40.00, 40.0, 41], 'child_process_ext_level': [1, 1.04, 1.0, 3],
    'child_process_integrity': [3, 5.3, 6.0, 6], 'child_process_md5': [32, 32.0, 32.0, 32],
    'child_process_name': [1, 13.54, 27.0, 163], 'child_process_net_level': [1, 1.99, 2.0, 2],
    'child_process_orig_name': [1, 13.06, 23.0, 63], 'child_process_path': [8, 42.91, 88.0, 174],
    'child_process_pdb': [6, 22.16, 60.0, 210], 'child_process_product_name': [1, 31.83, 36.0, 127],
    'child_process_service': [1, 13.07, 33.0, 156], 'child_process_signer': [2, 20.06, 27.0, 68],
    'command_line': [0, 79.77, 269.0, 20638], 'control': [6, 6.0, 6.0, 6], 
    'data_size': [1, 1.92, 4.0, 7], 'device_reserved': [1, 1.0, 1.0, 1],
    'device_type': [1, 1.0, 1.0, 1], 'dmpinfo': [34, 34.67, 35.8, 36],
    'edrver': [11, 11.0, 11.0, 11], 'event_cost': [3, 3.0, 3.0, 3],
    'file_action': [17, 26.97, 31.0, 31], 'file_cert_hash': [40, 40.00, 40.0, 41],
    'file_delete_count': [1, 1.04, 1.0, 3], 'file_device': [2, 2.00, 2.0, 3],
    'file_dltool': [1, 1.00, 1.0, 4], 'file_ext_level': [1, 1.00, 1.0, 3],
    'file_hidden_data': [1, 1.0, 1.0, 1], 'file_internal_name': [1, 13.98, 21.0, 127],
    'file_md5': [32, 32.0, 32.0, 32], 'file_name': [1, 21.27, 44.0, 229],
    'file_net_level': [2, 2.0, 2.0, 2], 'file_network': [21, 34.78, 44.0, 44],
    'file_oldname': [4, 22.03, 40.0, 92], 'file_oldpath': [9, 69.85, 108.0, 163],
    'file_orig_name': [1, 14.84, 28.0, 63], 'file_path': [0, 71.71, 119.0, 259],
    'file_pdb': [4, 54.25, 95.0, 259], 'file_product_name': [1, 18.14, 36.0, 127],
    'file_signer': [2, 25.84, 33.0, 68], 'file_source': [0, 40.16, 51.0, 124],
    'file_written_count': [1, 1.36, 2.0, 3], 'finalexcep': [4, 4.0, 4.0, 4],
    'flags': [1, 1.0, 1.0, 1], 'from_etw_data': [1, 1.0, 1.0, 1],
    'guid': [38, 38.0, 38.0, 38], 'image_path': [9, 36.85, 52.0, 92],
    'inject_size': [1, 3.29, 4.0, 17], 'inject_target_address': [5, 8.20, 13.0, 15],
    'inject_type': [11, 17.71, 18.0, 20], 'invalid_sign': [1, 1.0, 1.0, 1],
    'lnk_cmd': [1, 28.47, 47.0, 190], 'lnk_exe': [0, 25.51, 46.0, 106],
    'lnk_name': [27, 90.55, 131.0, 131], 'lnk_path': [9, 71.24, 91.0, 113],
    'load_driver_type': [1, 1.0, 1.0, 1], 'logon_domain': [15, 15.0, 15.0, 15],
    'matched_filepath_list': [9, 82.55, 114.0, 10550], 'mem_load': [2, 2.0, 2.0, 2],
    'online': [2, 2.0, 2.0, 2], 'operation_type': [1, 2.43, 6.0, 6], 
    'parent_command_line': [1, 38.96, 59.0, 20638], 'parent_process_name': [2, 17.81, 48.0, 163],
    'parent_process_net_level': [2, 2.0, 2.0, 2], 'parent_process_path': [8, 53.07, 127.0, 174],
    'path_hijacking': [1, 1.0, 1.0, 1], 'port_number': [1, 1.0, 1.0, 1],
    'procaction_timestamp': [24, 24.0, 24.0, 24],'process_build_name': [1, 15.30, 45.0, 127],
    'process_cert_hash': [40, 40.00, 40.0, 41], 'process_exec_hijack': [1, 1.0, 1.0, 1],
    'process_ext_level': [1, 1.02, 1.0, 3], 'process_file_version': [1, 23.8, 38.0, 63],
    'process_input_handle': [14, 14.00, 14.0, 17], 'process_integrity': [0, 5.47, 6.0, 6],
    'process_link': [9, 140.82, 730.0, 2199], 'process_link_level': [2, 2.0, 2.0, 2],
    'process_md5': [32, 32.0, 32.0, 32], 'process_name': [1, 17.43, 43.0, 163],
    'process_net_level': [2, 2.0, 2.0, 2], 'process_orig_name': [1, 16.52, 41.0, 63],
    'process_output_handle': [14, 14.00, 14.0, 17], 'process_path': [6, 59.38, 140.0, 174],
    'process_pdb': [6, 33.70, 90.0, 210], 'process_product_name': [1, 27.50, 36.0, 127],
    'process_readed_ext': [0, 6.39, 12.0, 84], 'process_remote_shell': [1, 1.0, 1.0, 1],
    'process_service': [1, 40.95, 168.0, 183], 'process_signer': [2, 20.28, 27.0, 68],
    'process_start_time': [10, 10.0, 10.0, 10], 'read_file_count': [1, 1.00, 1.0, 4],
    'read_file_io_counters': [1, 3.79, 5.0, 6], 'read_offset': [1, 10.83, 15.0, 15],
    'reg_name': [0, 6.43, 16.0, 64], 'reg_path': [23, 82.0, 115.0, 154], 
    'reg_previous_data': [1, 21.48, 38.0, 2048], 'reg_type': [0, 7.05, 10.0, 20],
    'reg_value': [0, 296.23, 3396.0, 3952], 'remote_ip': [12, 20.59, 35.0, 36],
    'remote_operate': [1, 1.0, 1.0, 1], 'remote_port': [3, 4.94, 5.0, 5], 
    'reparse_tag': [10, 10.0, 10.0, 10], 'reset_code': [10, 10.0, 10.0, 10], 
    'reset_info': [11, 11.0, 11.0, 11], 'reset_time': [1, 1.0, 1.0, 1], 
    'sensor_modules': [793, 1019.31, 1052.0, 1052], 'service_name': [1, 4.93, 16.0, 51], 
    'service_type': [1, 1.0, 1.0, 1], 'shell_code': [64, 64.0, 64.0, 64], 
    'shell_name': [13, 13.63, 14.0, 14], 'source_process_integrity': [0, 5.67, 6.0, 6], 
    'source_process_path': [10, 10.0, 10.0, 10], 'source_process_starttime': [10, 10.0, 10.0, 10],
    'target_command_line': [1, 10.87, 11.0, 3686], 'target_count': [1, 1.51, 2.0, 3], 
    'target_file_cert_hash': [40, 40.0, 40.0, 40], 'target_file_ext_level': [1, 1.03, 1.0, 3], 
    'target_file_md5': [32, 32.0, 32.0, 32], 'target_file_net_level': [2, 2.0, 2.0, 2],
    'target_file_path': [5, 60.32, 73.0, 143], 'target_file_pdb': [7, 39.37, 50.0, 135],
    'target_file_signer': [4, 21.06, 21.0, 59], 'target_process_cert_hash': [40, 40.0, 40.0, 40],
    'target_process_ext_level': [1, 1.042, 1.0, 3], 'target_process_md5': [0, 32.0, 32.0, 32],
    'target_process_name': [1, 12.51, 23.0, 163], 'target_process_net_level': [2, 2.0, 2.0, 2],
    'target_process_path': [8, 39.27, 102.0, 174], 'target_process_pdb': [6, 14.74, 32.0, 210],
    'target_process_signer': [2, 24.65, 27.0, 64], 'task_author': [3, 20.66, 23.0, 23],
    'task_name': [2, 112.65, 115.0, 115], 'thread_module': [2, 11.90, 29.0, 163], 
    'unicode_rtlo': [1, 1.0, 1.0, 4], 'virus_name': [20, 21.0, 22.0, 22], 
    'write_offset': [1, 9.56, 13.0, 13], 'zero_day_check': [3, 3.0, 3.0, 3], 
    'zero_day_index': [1, 1.0, 1.0, 1], 'zero_day_op': [5, 5.0, 5.0, 5], 
    'zero_day_type': [3, 3.34, 4.0, 5], 'event_type': [7, 11.81, 15.0, 18],
    'host_name': [15, 15.0, 15.0, 15],
    }
list_key_dims = [len(k)+int(np.ceil(v[2])) for k, v in used_keys.items()]
# 11643



def init_embed_tokens(model, tokenizer, new_tokens):
    for token in new_tokens:
        tokenized_old = tokenizer(token.strip("<").strip(">"), add_special_tokens=False)
        embeded = model.model.embed_tokens(torch.tensor(tokenized_old["input_ids"]))
        tokenized_new = tokenizer(token, add_special_tokens=False)
        model.model.embed_tokens.weight.data[tokenized_new["input_ids"][0],:]=embeded.mean(dim=0).data
    return model



def most_similar_string_idx(a, candidates):
    string = max(candidates, key=lambda x: SequenceMatcher(None, a, x).ratio())
    idx = candidates.index(string)
    return string, idx



def readfile(file_path, true_idx, time_len, max_time_len):
    is_training=False; ae=None; ac=None
    
    if max_time_len>0 and time_len>max_time_len:
        if len(true_idx)>0 and len(true_idx)>max_time_len:
            true_idx = random.sample(true_idx, max_time_len); usedidx = true_idx
            sorted(usedidx); sorted(true_idx)
        if len(true_idx)>0 and len(true_idx)<=max_time_len:
            pool_list = list(set(range(time_len))-set(true_idx))
            usedidx = true_idx+random.sample(pool_list, max_time_len-len(true_idx))
            sorted(usedidx); sorted(true_idx)
        if len(true_idx)==0:
            pool_list = list(range(time_len))
            usedidx = random.sample(pool_list, max_time_len)
            sorted(usedidx); true_idx=[]
    else:
        usedidx = list(range(time_len)); true_idx = true_idx
                    
    all_line_values = []; all_line_masks = []; flag=-1; ref_data=[]
    with open(file_path, 'r', encoding='utf-8', errors="ignore") as json_file:
        for log_line in json_file:
            try:
                data = json.loads(log_line)
            except json.JSONDecodeError as e:
                continue
            flag+=1
            if flag not in usedidx:
                continue
            data["event"] = {k: v for k, v in data["event"].items() if k in used_keys}
            data["meta"] = {k: v for k, v in data["meta"].items() if k in used_keys}
            if flag not in true_idx:
                ref_data.append(data["event"])
            if is_training and ae is not None:
                data["event"] = ae.apply(data["event"], random.choice(ref_data) if len(ref_data)>0 else {})
            
            #初始化 #line_values:[[], [], [], ..., []] line_masks: [0/1, 0/1, 0/1, ..., 0/1]
            line_values = []; line_masks = []
            for key in used_keys.keys():
                line_values.append(np.zeros(len(key)+int(np.ceil(used_keys[key][2])), dtype=np.uint8))
            line_masks = np.zeros(len(used_keys), dtype=np.uint8)
            for key in data["event"].keys():
                string, idx = most_similar_string_idx(key, list(used_keys.keys()))
                key_len = len(string); value_len = int(np.ceil(used_keys[string][2]))
                
                value_string = ''.join([char for char in str(data["event"][key]) if char.isascii()])
                input_string = key[-key_len:].ljust(key_len, "\0")+value_string[-value_len:].ljust(value_len, "\0")
                line_values[idx] = np.frombuffer(input_string.encode('utf-8'), dtype=np.uint8)
                line_masks[idx] = 1
            for key in data["meta"].keys():
                if key in data["event"].keys():
                    continue
                string, idx = most_similar_string_idx(key, list(used_keys.keys()))
                key_len = len(string); value_len = int(np.ceil(used_keys[string][2]))
                
                value_string = ''.join([char for char in str(data["meta"][key]) if char.isascii()])
                input_string = key[-key_len:].ljust(key_len, "\0")+value_string[-value_len:].ljust(value_len, "\0")
                line_values[idx] = np.frombuffer(input_string.encode('utf-8'), dtype=np.uint8)
                line_masks[idx] = 1
            all_line_values.append(line_values)
            all_line_masks.append(line_masks)
        if is_training and ac is not None:
            Allvalue, Truevalue = ac.apply(usedidx, true_idx)
            idxs = [usedidx.index(v) for v in Allvalue]
            all_line_values = [all_line_values[idx] for idx in idxs]
            all_line_masks = [all_line_masks[idx] for idx in idxs]
    #return all_line_values, all_line_masks
    data_save = {"all_line_values": all_line_values, "all_line_masks": all_line_masks}
    return data_save



    

    

######for train test
@dataclass
class CustomDataCollator:
    def __init__(self, tokenizer, system_prompt, template_prompt, max_time_len, is_training, is_ae, is_ac, each_p, family_tokens, datasets_path, datasets_name, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.template_prompt = template_prompt
        self.max_time_len = max_time_len
        self.is_training = True if is_training==1 else False
        self.ae = Augmentation_Event(overall_p=0.5, each_p=each_p) if is_ae==1 else None
        self.ac = Augmentation_Chain(overall_p=0.5, each_p=each_p) if is_ac==1 else None
        self.family_tokens = " ".join(family_tokens)
        self.datasets_path = datasets_path
        self.datasets_name = datasets_name
        self.config = config

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        familys = [feature['family'] for feature in features]
        file_paths = [feature['file_path'] for feature in features]
        true_idxs = [feature['true_idx'] for feature in features]
        time_lens = [feature['time_len'] for feature in features]

        return_batch = {}; text_segs_all_values=[]; text_segs_all_mask=[]; text_segs_num = []; messages=[]
        for i in range(len(familys)):
            # if we used get_npz_convenient to get .pkl.gz
            file_path = os.path.normpath(os.path.join(self.datasets_path.replace("Data_lists_cleaned/", ""), file_paths[i]+".pkl.gz"))
            with gzip.open(file_path, "rb") as f:
                loaded_data = pickle.load(f)
                all_line_values = loaded_data["all_line_values"]
                all_line_masks = loaded_data["all_line_masks"]
            ## else we read from meta data
            #file_path = os.path.normpath(os.path.join(self.datasets_path.replace("Data_lists_cleaned/", ""), file_paths[i]))
            #loaded_data = readfile(file_path, true_idxs[i], time_lens[i], self.max_time_len)
            #all_line_values = loaded_data["all_line_values"]
            #all_line_masks = loaded_data["all_line_masks"]
            if self.is_training and self.ae is not None:
                all_line_values, all_line_masks = self.ae.apply(all_line_values, all_line_masks)
            if self.is_training and self.ac is not None:
                 all_line_values, all_line_masks = self.ac.apply(all_line_values, all_line_masks)
            text_segs_all_values += all_line_values
            text_segs_all_mask += all_line_masks
            text_segs_num.append(len(all_line_values))
            
            _prompt = self.template_prompt.replace("{input}", "<|event_patch|>"*self.config.topK)
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _prompt},
            ]
            messages.append(message)
        
        if self.is_training:
            ####tokenizer
            texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts_tokenized = self.tokenizer(texts, add_special_tokens=False)
            answers_tokenized = self.tokenizer(familys, add_special_tokens=False)
            list_all_values = []; list_all_mask = []
            for k in range(len(list_key_dims)):
                list_all_values.append(torch.from_numpy(np.stack([item[k] for item in text_segs_all_values], axis=0)).to(dtype=torch.bfloat16))
                list_all_mask.append(torch.from_numpy(np.array([item[k] for item in text_segs_all_mask])).to(dtype=torch.bfloat16))
                
            ####concat
            return_batch = {}
            return_batch["input_ids"] = [texts_tokenized["input_ids"][i] + answers_tokenized["input_ids"][i] + [self.tokenizer.pad_token_id] for i in range(len(familys))]
            return_batch["attention_mask"] = [texts_tokenized["attention_mask"][i] + answers_tokenized["attention_mask"][i] + [1] for i in range(len(familys))]
            return_batch["labels"] = [[-100]*len(texts_tokenized["input_ids"][i]) + answers_tokenized["input_ids"][i] + [self.tokenizer.pad_token_id] for i in range(len(familys))]
            ####batch
            key_padids = {"input_ids":self.tokenizer.pad_token_id, "attention_mask":0, "labels":-100}
            MAX = max([len(return_batch["input_ids"][i]) for i in range(len(familys))])
            for key in key_padids.keys():
                return_batch[key] = torch.tensor([[key_padids[key]]*(MAX-len(return_batch[key][i])) + return_batch[key][i] for i in range(len(familys))])
            
            return_batch["list_all_values"] = list_all_values
            return_batch["list_all_mask"] = list_all_mask
            return_batch["segs_num"] = torch.tensor(text_segs_num)
            patch_token_id = self.tokenizer("<|event_patch|>", add_special_tokens=False)["input_ids"][0]
            is_patch = (return_batch["input_ids"] == patch_token_id).to(torch.int)
            start = is_patch.argmax(dim=1)  #for dim=1, get the first True
            end = is_patch.flip(dims=[1]).argmax(dim=1); end = (is_patch.shape[-1] - end - 1)
            return_batch["segs_se"] = torch.stack((start, end), dim=1)
            return return_batch
        else:
            ####tokenizer
            texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts_tokenized = self.tokenizer(texts, add_special_tokens=False)
            list_all_values = []; list_all_mask = []
            for k in range(len(list_key_dims)):
                list_all_values.append(torch.from_numpy(np.stack([item[k] for item in text_segs_all_values], axis=0)).to(dtype=torch.bfloat16))
                list_all_mask.append(torch.from_numpy(np.array([item[k] for item in text_segs_all_mask])).to(dtype=torch.bfloat16))
            ####concat
            return_batch = {}
            return_batch["input_ids"] = [texts_tokenized["input_ids"][i] for i in range(len(familys))]
            return_batch["attention_mask"] = [texts_tokenized["attention_mask"][i] for i in range(len(familys))]
            ####batch
            key_padids = {"input_ids":self.tokenizer.pad_token_id, "attention_mask":0}
            MAX = max([len(return_batch["input_ids"][i]) for i in range(len(familys))])
            for key in key_padids.keys():
                return_batch[key] = torch.tensor([[key_padids[key]]*(MAX-len(return_batch[key][i])) + return_batch[key][i] for i in range(len(familys))])

            return_batch["list_all_values"] = list_all_values
            return_batch["list_all_mask"] = list_all_mask
            return_batch["segs_num"] = torch.tensor(text_segs_num)
            patch_token_id = self.tokenizer("<|event_patch|>", add_special_tokens=False)["input_ids"][0]
            is_patch = (return_batch["input_ids"] == patch_token_id).to(torch.int)
            start = is_patch.argmax(dim=1)  #for dim=1, get the first True
            end = is_patch.flip(dims=[1]).argmax(dim=1); end = (is_patch.shape[-1] - end - 1)
            return_batch["segs_se"] = torch.stack((start, end), dim=1)
            return return_batch, features
    
        

class JSONLoggerCallback(TrainerCallback):
    def __init__(self, log_file, save_steps):
        self.log_file = log_file
        self.save_steps = save_steps
        self.logs = []
    def on_log(self, args, state, control, **kwargs):
        log_entry = {
            "global_step": state.global_step,
            "logs": kwargs['logs'],
        }
        self.logs.append(log_entry)
        if state.global_step % self.save_steps == 0:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=4)