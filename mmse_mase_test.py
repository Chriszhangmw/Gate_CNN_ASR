import torch
from models.conv import GatedConv
import os
import json
import hashlib
import base64
import hmac
import json
import os


def get_test_by_mase(path):
    model = GatedConv.load("/home/zmw/big_space/zhangmeiwei_space/asr_res_model/masr/ad/model_best_finture.pth")
    device = torch.device("cuda:3")
    torch.cuda.set_device(device)
    total_num = 0
    catalog_num = 0
    text_dic = {}
    dirs = os.listdir(path)
    for dir in dirs:
        patient_catalog = {}
        patient_name = str(dir)
        child_path = os.path.join(path,dir)
        child_files = os.listdir(child_path)
        for file in child_files:
            if file.endswith('wav'):
                total_num += 1
                file_catalogs = str(file).replace('.wav','')
                file_catalogs = file_catalogs.split('Z')
                catalog = file_catalogs[1]
                print(catalog)
                catalog_num+=1
                read_path = os.path.join(child_path,file)
                text = model.predict(read_path)
                print(text)
                patient_catalog[catalog] = text
        text_dic[patient_name] = patient_catalog
    print("total sample :",total_num,catalog_num)
    with open('./mmse/asr_f2.json','w',encoding='utf-8') as f:
        json.dump(text_dic,f,indent=2,sort_keys=True,ensure_ascii=False)






if __name__ == '__main__':
    path = '/home/zmw/big_space/zhangmeiwei_space/AD_TEST/v_20211125/dd'
    get_test_by_mase(path)







