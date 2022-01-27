
import torch
import os
import math
import torch.nn as nn
from models.conv import GatedConv


def save_model(model, epoch):
    save_path = "/home/zmw/big_space/zhangmeiwei_space/asr_res_model/masr/aishell/model_best.pth"
    print("SAVE MODEL to", save_path)
    args = {
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }
    torch.save(args, save_path)



def load_model(vocabulary):
    path = "/home/zmw/big_space/zhangmeiwei_space/asr_res_model/masr/aishell/model_best_drop.pth"
    checkpoint = torch.load(path, map_location=torch.device('cuda:4'))
    # print(checkpoint.keys())
    # epoch = checkpoint['epoch']
    return checkpoint
    # model = GatedConv(vocabulary)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # return model, epoch
















