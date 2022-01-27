
from models.conv import GatedConv
import torch
import Levenshtein as Lev

model = GatedConv.load("/home/zmw/big_space/zhangmeiwei_space/asr_res_model/masr/ad/model_best_drop2_a0.1.pth")
device = torch.device("cuda:2")
torch.cuda.set_device(device)
# trainFile = "/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/train.csv"
# valFile = "/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/dev.csv"
testFile = "/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/test.csv"

# trainFile_new = open("/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/train_bert.csv",'w',encoding='utf-8')
# valFile_new = open("/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/dev_bert.csv",'w',encoding='utf-8')
testFile_new = open("/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/dev_bert.csv",'w',encoding='utf-8')

num = 0
# with open(trainFile,'r',encoding='utf-8') as f:
#     train = f.readlines()
# for line in train:
#     print('processing training: ',num)
#     line = line.strip().split(',')
#     label = line[1]
#     p = line[0]
#     text = model.predict(p)
#     trainFile_new.write(label+','+text+'\n')
#     num += 1

# with open(valFile,'r',encoding='utf-8') as f:
#     val = f.readlines()
# for line in val:
#     print('processing validating: ', num)
#     line = line.strip().split(',')
#     label = line[1]
#     p = line[0]
#     text = model.predict(p)
#     valFile_new.write(label+','+text+'\n')
#     num += 1

with open(testFile,'r',encoding='utf-8') as f:
    val = f.readlines()
for line in val:
    print('processing validating: ', num)
    line = line.strip().split(',')
    label = line[1]
    p = line[0]
    text = model.predict(p)
    testFile_new.write(label+','+text+'\n')
    num += 1











