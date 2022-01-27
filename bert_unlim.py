import warnings
warnings.filterwarnings("ignore")
import os

import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from tensorflow.keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import os
os.environ['TF_KERAS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 基本参数
maxlen = 64
batch_size = 32
epochs = 20

# bert配置
config_path = '/home/zmw/big_space/zhangmeiwei_space/pre_models/tensorflow/publish/bert_config.json'
checkpoint_path = '/home/zmw/big_space/zhangmeiwei_space/pre_models/tensorflow/publish/bert_model.ckpt'
dict_path = '/home/zmw/big_space/zhangmeiwei_space/pre_models/tensorflow/publish/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            label,input = l.strip().split(',')
            if input == '':
                continue
            D.append((input,label))
    return D


# 加载数据集
train_data = load_data('/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/train_bert.csv')
valid_data = load_data('/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/dev_bert.csv')


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (input,label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                input, label, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs) #这里的2表示输出的维度是多少

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights('./bert/best_model.weights')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for (content,title) in tqdm(data):
            title = ' '.join(title)
            pred_title = ' '.join(autotitle.generate(content, topk))
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[title.split(' ')],
                    hypothesis=pred_title.split(' '),
                    smoothing_function=self.smooth
                )
                total += 1
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }



import Levenshtein as Lev
def cal_cer(s1, s2):
    s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
    return Lev.distance(s1, s2)
from matrix import calc_f1,bleu,distinct

if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    # with open('/home/zmw/big_space/zhangmeiwei_space/asr_data/cleaed_data/data_ad/dev_bert.csv', encoding='utf-8') as f:
    #     testdata = f.readlines()
    # model.load_weights('./bert/best_model.weights')
    # cer = 0
    # num = 0
    # cer1 = 0
    # F1data = []
    # predictions = []
    #
    # for line in testdata:
    #     line = line.strip().split(',')
    #     label= line[0]
    #     line_x = line[1]
    #     topk = 1
    #     pred_text = autotitle.generate(line_x, topk)
    #     v1 = float(cal_cer(pred_text, label))
    #     cer += v1 / float(len(label))
    #     num += 1
    #     F1data.append((pred_text, label))
    #     predictions.append(pred_text)
    # f1 = calc_f1(F1data)
    # bleu_1, bleu_2 = bleu(F1data)
    # unigrams_distinct, bigrams_distinct, intra_dist1, intra_dist2 = distinct(predictions)
    # print(f1, bleu_1, bleu_2, unigrams_distinct, bigrams_distinct, intra_dist1, intra_dist2)
    # cer = cer / float(num)
    # print("测试集上的CER : ", cer)



# else:
#
#     model.load_weights('./best_model.weights')