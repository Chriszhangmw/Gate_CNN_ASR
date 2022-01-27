
import tqdm
from models.conv import GatedConv
import torch
import Levenshtein as Lev
from decoder import GreedyDecoder
import joblib

from matrix import calc_f1,bleu,distinct

def cal_cer(s1, s2):
    s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
    return Lev.distance(s1, s2)

def cal_test(model,testIndexPath):
    cer2 = 0
    with open(testIndexPath, 'r', encoding='utf-8') as f:
        idx = f.readlines()
    for line in idx:
        line = line.strip().split(',')
        path = line[0]
        label = line[1]
        text = model.predict(path)
        v1 = float(cal_cer(text, label))
        cer2 += v1 / float(len(label))
    cer2 = cer2 / float(len(idx))
    print("测试集上的CER : ", cer2)


def writetrain2bert(model,trainFile):
    new_train = open('./data_aishell/train.csv','w',encoding='utf-8')
    with open(trainFile,'r',encoding='utf-8') as f:
        data = f.readlines()
        f.close()
    for line in data:
        line = line.strip().split(',')
        label = line[1]
        p = line[0]
        text = model.predict(p)
        new_train.write(label + ',' + text + '\n')

def writetest2bert(model,testFile):
    new_train = open('./data_aishell/test.csv','w',encoding='utf-8')
    with open(testFile,'r',encoding='utf-8') as f:
        data = f.readlines()
        f.close()
    for line in data:
        line = line.strip().split(',')
        label = line[1]
        p = line[0]
        text = model.predict(p)
        new_train.write(label + ',' + text + '\n')

from matrix import calc_f1,bleu,distinct


def cer(s1, s2):
    s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
    return Lev.distance(s1, s2)



if __name__ == "__main__":
    model = GatedConv.load("/home/zmw/big_space/zhangmeiwei_space/asr_res_model/masr/ad/model_best_drop2_a0.1.pth")
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)
    testFile = "/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/test.csv"
    # cal_test(model,testFile)
    # res = open('./zmw_result.csv','w',encoding='utf-8')
    with open(testFile,'r',encoding='utf-8') as f:
        data = f.readlines()
    number = 0

    cer1 = 0
    F1data = []
    predictions = []

    for line in data:
        line = line.strip().split(',')
        label = line[1]
        p = line[0]
        text = model.predict(p)
        temp = float(cer(text, label))
        temp = temp/len(label)
        cer1 += temp
        F1data.append((text, label))
        predictions.append(text)
        number += 1
    cer1 = cer1/number
    print(cer1)
    f1 = calc_f1(F1data)
    bleu_1, bleu_2 = bleu(F1data)
    unigrams_distinct, bigrams_distinct, intra_dist1, intra_dist2 = distinct(predictions)
    print(f1, bleu_1, bleu_2, unigrams_distinct, bigrams_distinct, intra_dist1, intra_dist2)















