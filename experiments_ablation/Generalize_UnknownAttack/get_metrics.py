import sys
import io
import os
import json
import random
import numpy as np
import concurrent.futures
import multiprocessing
from collections import Counter
import matplotlib.pyplot as plt




def calcute_recall_false_alarm(ground, predict):
    family_tokens = list(set(ground))
    family_DICT = {}
    for family in family_tokens:
        if family not in family_DICT:
            family_DICT[family] = []
    white_tokens = ["<white>"]
    family_tokens.remove("<white>"); virus_tokens = family_tokens
    
    ####binary_acc
    binary_acc = (sum(1 for x, y in zip(ground, predict) if x not in virus_tokens and y not in virus_tokens) + sum(1 for x, y in zip(ground, predict) if x in virus_tokens and y in virus_tokens)) / len(ground)
    
    ####recall false_alarm 
    rec=0.0; total1=0.0;  alarm = 0.0; total2 = 0.0;  
    for gro, pre in zip(ground, predict):
        if gro!="<white>":
            total1 += 1
            if pre in virus_tokens:
                rec += 1
        if gro=="<white>":
            total2 += 1
            if pre in virus_tokens:
                alarm += 1
    recall = rec/total1
    false_alarm = alarm/total2
    
    ####overall_acc each_acc
    for gro, pre in zip(ground, predict):
        if gro=="<white>":
            if pre not in virus_tokens:
                family_DICT[gro].append(1)
            else:
                family_DICT[gro].append(0)
        else:
            if gro==pre:
                family_DICT[gro].append(1)
            else:
                family_DICT[gro].append(0)
    each_acc={}; get=0.0; total3=0.0
    for family in family_DICT.keys():
        each_acc[family] = sum(family_DICT[family])/len(family_DICT[family])
        get += sum(family_DICT[family])
        total3 += len(family_DICT[family])
    overall_acc = get/total3
    return binary_acc, recall, false_alarm, overall_acc, each_acc







####"test_dir_paramS48HY4/@30000forUnknownAttack"
with open(os.path.join("../../datasets/Data_lists_cleaned/", "Families.json"), "r", encoding="utf-8") as file:
    families = json.load(file)
    families = ["<"+item+">" for item in families]
    families.remove("<white>")
fold = "test_dir_paramS48HY4/@30000forUnknownAttack"
with open(os.path.join(fold, "results_file.txt"), "r", encoding="utf-8") as file:
    data = json.load(file)
    for gro, pre in zip(data["ground"], data["predict"]):
        if gro=="<white>":
            ground_w = gro
            predict_w = "<unknown>" if pre in families else "<white>"
        else:
            ground_u = "<unknown>"
            predict_u = "<unknown>" if pre in families else "<white>"
binary_acc_list = []; recall_list = []; false_alarm_list =[]
for i in range(10):
    random_pairs = random.sample(list(zip(ground_u, predict_u)), k=1000)
    selected_list1, selected_list2 = zip(*random_pairs)
    ground = ground_w + list(selected_list1)
    predict = predict_w + list(selected_list2)
    binary_acc, recall, false_alarm, overall_acc, each_acc = calcute_recall_false_alarm(ground, predict)
    binary_acc_list.append(binary_acc)
    recall_list.append(recall)
    false_alarm_list.append(false_alarm)
print(f"{fold}: binary_acc: {sum(binary_acc_list)/10}. recall: {sum(recall_list)/10}. false_alarm: {sum(false_alarm_list)/10}.")
