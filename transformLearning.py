import torch
import torch.nn as nn
import data
from models.conv_transformerLearning import GatedConv
# import models.conv as con
# import models
from tqdm import tqdm
from decoder import GreedyDecoder
from warpctc_pytorch import CTCLoss
import torch.nn.functional as F
import joblib
from matrix import calc_f1,bleu,distinct
from save_model import load_model


new_word_ad = [w.strip() for w in open('./newword.csv','r',encoding='utf-8').readlines()]

def kloss(en1,en2):
    # print('en1.shape:', en1.shape)
    # print('en1.shape:', en1.shape)
    # print('F.log_softmax(en2, dim=-1) ',F.log_softmax(en2, dim=-1).shape)
    kl_loss1 = F.kl_div(F.log_softmax(en1, dim=-1), F.softmax(en2, dim=-1), reduction='sum')
    kl_loss1 = kl_loss1.sum()
    kl_loss2 = F.kl_div(F.log_softmax(en2, dim=-1), F.softmax(en1, dim=-1), reduction='sum')
    kl_loss2 = kl_loss2.sum()
    kl_loss = (kl_loss1 + kl_loss2)/2
    return kl_loss


def train(
    model1,
    model2,
    start_epoch,
    learning_rate,
    epochs=500,
    batch_size=4,
    train_index_path="/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/train.csv",
    dev_index_path="/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/dev.csv",
    test_index_path = "/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/test.csv",
    labels_path="./data_aishell/labels.gz",
    momentum=0.8,
    max_grad_norm=0.2,
    weight_decay=0
):
    train_dataset = data.MASRDataset(train_index_path, labels_path)
    batchs = (len(train_dataset) + batch_size - 1) // batch_size
    dev_dataset = data.MASRDataset(dev_index_path, labels_path)
    test_dataset = data.MASRDataset(test_index_path, labels_path)
    train_dataloader = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8
    )
    train_dataloader_shuffle = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True
    )
    dev_dataloader = data.MASRDataLoader(
        dev_dataset, batch_size=batch_size, num_workers=8
    )
    test_dataloader = data.MASRDataLoader(
        test_dataset, batch_size=batch_size, num_workers=8
    )
    # orignal data
    train_index_path2 = '/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_thch30/train.csv'
    train_dataset2 = data.MASRDataset(train_index_path2, labels_path)
    batchs2 = (len(train_dataset2) + batch_size - 1) // batch_size
    train_dataloader2 = data.MASRDataLoader(
        train_dataset2, batch_size=batch_size, num_workers=8
    )
    train_dataloader_shuffle2 = data.MASRDataLoader(
        train_dataset2, batch_size=batch_size, num_workers=8, shuffle=True
    )

    # parameters1 = model1.parameters()
    # optimizer1 = torch.optim.SGD(
    #     parameters1,
    #     lr=learning_rate,
    #     momentum=momentum,
    #     nesterov=True,
    #     weight_decay=weight_decay,
    # )
    parameters2 = model2.parameters()
    optimizer2 = torch.optim.SGD(
        parameters2,
        lr=learning_rate,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )
    ctcloss = CTCLoss(size_average=True)
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer2, 0.985)
    # writer = tensorboard.SummaryWriter()
    gstep = 0
    best_cer = 100000
    best_cer2 = 100000
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        if epoch > 0:
            train_dataloader = train_dataloader_shuffle
            train_dataloader2 = train_dataloader_shuffle2
        # for orignal,target in zip(train_dataloader,train_dataloader2):
        total = zip(train_dataloader,train_dataloader2)
        for i,inputs in enumerate(total):
            (x, y, x_lens, y_lens),(x2, y2, x_lens2, y_lens2) = inputs
        # for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
        #     x2, y2, x_lens2, y_lens2 = train_dataloader2[i]
            x = x.to("cuda")
            x2 = x2.to("cuda")
            out1, out_lens1 = model1(x, x_lens)
            out1 = out1.transpose(0, 1).transpose(0, 2)
            loss1 = ctcloss(out1, y, out_lens1, y_lens)
            loss1 = loss1.to("cuda")

            out2, out_lens2 = model2(x, x_lens)
            out2 = out2.transpose(0, 1).transpose(0, 2)
            loss2 = ctcloss(out2, y, out_lens2, y_lens)
            loss2 = loss2.to("cuda")

            out3, out_lens3 = model2(x2, x_lens2)
            out3 = out3.transpose(0, 1).transpose(0, 2)
            loss3 = ctcloss(out3, y2, out_lens3, y_lens2)
            loss3 = loss3.to("cuda")

            s_loss1 = kloss(out1,out2)
            s_loss1 = s_loss1.to("cuda")

            Loss1 = 0.3 * s_loss1 + (1-0.3)*loss2

            loss = 0.4 * Loss1 + 0.6 * loss3

            loss = loss.to("cuda")

            # optimizer2.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model2.parameters(), max_grad_norm)
            optimizer2.step()
            # lr_sched.step()
            model2.zero_grad()
            epoch_loss += loss.item()
            gstep += 1
            print(
                "[{}/{}][{}/{}]\tLoss = {}".format(
                    epoch + 1, epochs, i, int(batchs), loss.item()
                )
            )
        epoch_loss = epoch_loss / batchs
        cer = eval(model2, dev_dataloader,epoch)
        print("Epoch {}: Loss= {}, CER = {}".format(epoch, epoch_loss, cer))
        if cer < best_cer:
            torch.save(model2, "/home/zmw/big_space/zhangmeiwei_space/asr_res_model/masr/ad/model_best_tl.pth")
            best_cer = cer
        if epoch > 100:
            test_cer = eval_test(model2, test_dataloader)
            if test_cer < best_cer2:
                best_cer2 = test_cer
                print("Best CER on Testing is : ", best_cer2)


def eval_test(model,dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
    F1data = []
    predictions = []
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            x = x.to("cuda")
            outs, out_lens = model(x, x_lens)
            outs = F.softmax(outs, 1)
            outs = outs.transpose(1, 2)
            ys = []
            offset = 0
            for y_len in y_lens:
                ys.append(y[offset: offset + y_len])
                offset += y_len
            out_strings, out_offsets = decoder.decode(outs, out_lens)
            y_strings = decoder.convert_to_strings(ys)
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                temp = float(decoder.cer(trans, ref))
                F1data.append((trans, ref))
                predictions.append(trans)
                cer += temp / float(len(ref))
        cer /= len(dataloader.dataset)
    f1 = calc_f1(F1data)
    bleu_1, bleu_2 = bleu(F1data)
    unigrams_distinct, bigrams_distinct, intra_dist1, intra_dist2 = distinct(predictions)
    print(f1, bleu_1, bleu_2, unigrams_distinct, bigrams_distinct, intra_dist1, intra_dist2)
    model.train()
    print("在测试集上的CER值为： ",cer)
    return cer


def eval(model, dataloader,epoch):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
    print("decoding")
    F1data = []
    predictions = []
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            x = x.to("cuda")
            outs, out_lens = model(x, x_lens)
            outs = F.softmax(outs, 1)
            outs = outs.transpose(1, 2)
            ys = []
            offset = 0
            for y_len in y_lens:
                ys.append(y[offset : offset + y_len])
                offset += y_len
            out_strings, out_offsets = decoder.decode(outs, out_lens)
            y_strings = decoder.convert_to_strings(ys)
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                temp = float(decoder.cer(trans, ref))
                F1data.append((trans, ref))
                predictions.append(trans)
                cer += temp / float(len(ref))
        cer /= len(dataloader.dataset)
    f1 = calc_f1(F1data)
    bleu_1, bleu_2 = bleu(F1data)
    unigrams_distinct, bigrams_distinct ,intra_dist1,intra_dist2= distinct(predictions)
    print(f1,bleu_1, bleu_2,unigrams_distinct, bigrams_distinct ,intra_dist1,intra_dist2)
    model.train()
    return cer


if __name__ == "__main__":
    vocabulary = joblib.load('./data_aishell/labels.gz')
    # print(vocabulary)
    vocabulary.extend(new_word_ad)
    vocabulary = "".join(vocabulary)
    continue_train = False
    if continue_train:
        model1,model2, epoch = load_model(vocabulary)
        start_epoch = epoch
        lr = 0.01
    else:
        model1 = GatedConv(vocabulary)
        model2 = GatedConv(vocabulary)
        start_epoch = 0
        lr = 0.6
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)
    model1.to(device)
    model2.to(device)
    train(model1,model2,start_epoch,lr)
