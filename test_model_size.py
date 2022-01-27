from thop import profile
from thop import clever_format





import torch

from models.conv import GatedConv

import joblib



new_word_ad = [w.strip() for w in open('./newword.csv','r',encoding='utf-8').readlines()]
x = torch.rand(4, 161, 511)
lens = torch.rand(4)
vocabulary = joblib.load('./data_aishell/labels.gz')
# print(vocabulary)
vocabulary.extend(new_word_ad)
vocabulary = "".join(vocabulary)
nnet = GatedConv(vocabulary)

Macs, params = profile(nnet, inputs=(x,lens))
Macs, params = clever_format([Macs, params], "%.3f")
print('Model Summary:')
print('Trainable params of the model is:', params)
print('MAC of the model is:', Macs)