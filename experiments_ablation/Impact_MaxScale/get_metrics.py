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






####"test_dir_paramS24HY4/@30000"
fold = "test_dir_paramS24HY4/@30000"
with open(os.path.join(fold, "results_file.txt"), "r", encoding="utf-8") as file:
    data = json.load(file)
    ground = data["ground"]
    predict = data["predict"]
binary_acc, recall, false_alarm, overall_acc, each_acc = calcute_recall_false_alarm(ground, predict)
print("========================================")
print(f"{fold}: binary_acc: {binary_acc}. recall: {recall}. false_alarm: {false_alarm}. overall_acc: {overall_acc}. each_acc:{each_acc}")


####"test_dir_paramS32HY4/@30000"
fold = "test_dir_paramS32HY4/@30000"
with open(os.path.join(fold, "results_file.txt"), "r", encoding="utf-8") as file:
    data = json.load(file)
    ground = data["ground"]
    predict = data["predict"]
binary_acc, recall, false_alarm, overall_acc, each_acc = calcute_recall_false_alarm(ground, predict)
print("========================================")
print(f"{fold}: binary_acc: {binary_acc}. recall: {recall}. false_alarm: {false_alarm}. overall_acc: {overall_acc}. each_acc:{each_acc}")


####"test_dir_paramS40HY4/@30000"
fold = "test_dir_paramS40HY4/@30000"
with open(os.path.join(fold, "results_file.txt"), "r", encoding="utf-8") as file:
    data = json.load(file)
    ground = data["ground"]
    predict = data["predict"]
binary_acc, recall, false_alarm, overall_acc, each_acc = calcute_recall_false_alarm(ground, predict)
print("========================================")
print(f"{fold}: binary_acc: {binary_acc}. recall: {recall}. false_alarm: {false_alarm}. overall_acc: {overall_acc}. each_acc:{each_acc}")


####"test_dir_paramS56HY4/@30000"
fold = "test_dir_paramS56HY4/@30000"
with open(os.path.join(fold, "results_file.txt"), "r", encoding="utf-8") as file:
    data = json.load(file)
    ground = data["ground"]
    predict = data["predict"]
binary_acc, recall, false_alarm, overall_acc, each_acc = calcute_recall_false_alarm(ground, predict)
print("========================================")
print(f"{fold}: binary_acc: {binary_acc}. recall: {recall}. false_alarm: {false_alarm}. overall_acc: {overall_acc}. each_acc:{each_acc}")