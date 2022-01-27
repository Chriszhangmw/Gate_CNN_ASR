
import os
import re
import joblib
import librosa
import torch
import wave
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from random import shuffle


base_root = './data_aishell'
train_path_dir = './data_aishell/train'
dev_path_dir = './data_aishell/dev'
test_path_dir = './data_aishell/test'


train_files_path = []
dev_files_path = []
test_files_path = []
words = []

_d = {}
with open('./data_aishell/aishell_transcript_v0.8.txt') as f0:
    data = f0.readlines()
    f0.close()
    for i in tqdm(data):
        k, v = re.split('\s+', i, 1)
        _d[k.strip()] = v.replace('\n', '').replace('\t', '').replace(' ', '')

def recur_train(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            train_files_path.append(os.path.join(root,file))
        for dir in dirs:
            recur_train(dir)
    res_train = []
    for file in tqdm(train_files_path):
        file_name = file.split('/')[-1][:-4]
        if file_name in _d:
            res_train.append((file, _d[file_name]))
    pd.DataFrame(res_train).to_csv('./data_aishell/train.index', index=False, header=None)

def recur_dev(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            dev_files_path.append(os.path.join(root,file))
        for dir in dirs:
            recur_dev(dir)
    res_dev = []
    for file in tqdm(dev_files_path):
        file_name = file.split('/')[-1][:-4]
        if file_name in _d:
            res_dev.append((file, _d[file_name]))
    pd.DataFrame(res_dev).to_csv('data_aishell/dev.index', index=False, header=None)

def recur_test(rootdir):
    for root, dirs, files in tqdm(os.walk(rootdir)):
        for file in files:
            if 'DS_Store' in file:
                continue
            test_files_path.append(os.path.join(root,file))
        for dir in dirs:
            recur_dev(dir)
    test_dev = []
    for file in tqdm(test_files_path):
        file_name = file.split('/')[-1][:-4]
        if file_name in _d:
            test_dev.append((file, _d[file_name]))
    pd.DataFrame(test_dev).to_csv('data_aishell/test.index', index=False, header=None)

# recur_train(train_path_dir)
# recur_dev(dev_path_dir)
# recur_test(test_path_dir)
# all_words = list(set(''.join([v for v in _d.values()])))
# all_words = ['_'] + all_words[:27] + [' '] + all_words[27:]
# joblib.dump(all_words, 'data_aishell/labels.gz')
# all_words = list(set(''.join([v for v in words])))
# all_words = ['_'] + all_words[:27] + [' '] + all_words[27:]
# print(len(all_words))
# joblib.dump(all_words, 'labels.gz')

# all_words = joblib.load('./data_aishell/labels.gz')
# print(all_words)
# print(len(all_words))




















