import re
import string
import numpy as np
import pandas as pd
import json
from math import ceil
import sys
import random




class Augmentation_Event():
    def __init__(self, overall_p=0.5, each_p=0.2):
        self.overall_p = overall_p
        self.each_p = each_p
    
    def event_kv_add(self, all_line_values, all_line_masks):
        ####all_line_values:[ [[...],[...],[...],...,[...]], [[...],[...],[...],...,[...]],... ]
        ####all_line_masks: [ [..........],[..........],... ]
        all_line_masks_temp = np.array(all_line_masks)
        row_indices, col_indices = np.where((all_line_masks_temp==0) & (np.random.rand(*all_line_masks_temp.shape) < self.each_p))
        for ridx, cidx in zip(row_indices, col_indices):
            row_rand = np.random.randint(0, len(all_line_values))
            all_line_values[ridx][cidx] = all_line_values[row_rand][cidx]
            all_line_masks[ridx][cidx] = all_line_masks[row_rand][cidx]
        
        return all_line_values, all_line_masks
    
    def event_k_revise(self, all_line_values, all_line_masks):
        ####all_line_values:[ [[...],[...],[...],...,[...]], [[...],[...],[...],...,[...]],... ]
        ####all_line_masks: [ [..........],[..........],... ]
        all_line_masks_temp = np.array(all_line_masks)
        row_indices, col_indices = np.where((all_line_masks_temp==1) & (np.random.rand(*all_line_masks_temp.shape) < self.each_p))
        for ridx, cidx in zip(row_indices, col_indices):
            array_temp = np.array(all_line_values[ridx][cidx].copy())
            flag = np.random.randint(0, 4)
            if flag==0:#Switch to lowercase a-z:97-122
                mask = (array_temp >= 97) & (array_temp <= 122) & (np.random.rand(*array_temp.shape)<self.each_p)
                array_temp[mask] -= 32
                all_line_values[ridx][cidx] = array_temp.tolist()
            if flag==1:#Switch to uppercase A-Z:65-90
                mask = (array_temp >= 65) & (array_temp <= 90) & (np.random.rand(*array_temp.shape)<self.each_p)
                array_temp[mask] += 32
                all_line_values[ridx][cidx] = array_temp.tolist()
            if flag==2:#Replace with spaces: 32
                mask = (np.random.rand(*array_temp.shape)<self.each_p)
                array_temp[mask] = 32
                all_line_values[ridx][cidx] = array_temp.tolist()
            if flag==3:#Replace with underline: 95
                mask = (np.random.rand(*array_temp.shape)<self.each_p)
                array_temp[mask] = 95
                all_line_values[ridx][cidx] = array_temp.tolist()
        return all_line_values, all_line_masks

    def apply(self, all_line_values, all_line_masks):
        if random.random()<self.overall_p:
            return all_line_values, all_line_masks
        function_dicts = {
            "event_kv_add": self.event_kv_add,
            "event_k_revise": self.event_k_revise,
        }
        items = list(function_dicts.items())
        random.shuffle(items)
        function_dicts = dict(items)
        Num = 1
        for i, key in enumerate(function_dicts.keys()):
            all_line_values, all_line_masks = function_dicts[key](all_line_values, all_line_masks)
            if i+1==Num:
                break
        return all_line_values, all_line_masks
# #######Check
# single_event = {"abc_abc": 1, "def_hi_jk_":2, "lm_n":3}
# ref_event = {"abc_abc": 100, "opq_rs":500, "tu_v_w":600}
# print(single_event)
# Augmentation_Event = Augmentation_Event(4, 0.3, 0.3)
# single_event = Augmentation_Event.apply(single_event, ref_event)
# print(single_event)

    


class Augmentation_Chain():
    def __init__(self, overall_p=0.5, each_p=0.2):
        self.overall_p = overall_p
        self.each_p = each_p

    def norm_event_remove(self, all_line_values, all_line_masks):
        ####all_line_values:[ [[...],[...],[...],...,[...]], [[...],[...],[...],...,[...]],... ]
        ####all_line_masks: [ [..........],[..........],... ]
        indices_to_remove = np.where(np.random.rand(len(all_line_values)) < self.each_p)[0].tolist()
        
        all_line_values = [x for i, x in enumerate(all_line_values) if i not in indices_to_remove]
        all_line_masks = [x for i, x in enumerate(all_line_masks) if i not in indices_to_remove]
        
        return all_line_values, all_line_masks
    
    def norm_event_add(self, all_line_values, all_line_masks):
        ####all_line_values:[ [[...],[...],[...],...,[...]], [[...],[...],[...],...,[...]],... ]
        ####all_line_masks: [ [..........],[..........],... ]
        indices_to_add = np.where(np.random.rand(len(all_line_values)) < self.each_p)[0].tolist()
        values_v = []; values_m = []
        for _ in indices_to_add:
            random_index = np.random.randint(0, len(all_line_values))
            values_v.append(all_line_values[random_index])
            values_m.append(all_line_masks[random_index])
        indices_to_add = indices_to_add[::-1]
        values_v = values_v[::-1]
        values_m = values_m[::-1]
        
        for idx, value_v, value_m in zip(indices_to_add, values_v, values_m):
            all_line_values.insert(idx, value_v)
            all_line_masks.insert(idx, value_m)
    
        return all_line_values, all_line_masks
        
    def apply(self, all_line_values, all_line_masks):
        if random.random()<self.overall_p:
            return all_line_values, all_line_masks
        function_dicts = {
            "norm_event_remove": self.norm_event_remove,
            "norm_event_add": self.norm_event_add,
        }
        items = list(function_dicts.items())
        random.shuffle(items)
        function_dicts = dict(items)
        Num = 1
        for i, key in enumerate(function_dicts.keys()):
            all_line_values, all_line_masks = function_dicts[key](all_line_values, all_line_masks)
            if i+1==Num:
                break   
        return all_line_values, all_line_masks
# #######Check
# Allvalue = [0,2,1,4,6,7,3,8,5]; Truevalue=[6,7,3,5]
# print(Allvalue, Truevalue)
# Augmentation_Chain = Augmentation_Chain(4, 0.9, 0.3)
# Allvalue, Truevalue = Augmentation_Chain.apply(Allvalue, Truevalue)
# print(Allvalue, Truevalue)