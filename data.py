import torch
import librosa
import wave
import numpy as np
import scipy
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import joblib


sample_rate = 16000
window_size = 0.02
window_stride = 0.01
n_fft = int(sample_rate * window_size)
win_length = n_fft
hop_length = int(sample_rate * window_stride)
window = "hamming"


def load_audio(wav_path, normalize=True):  # -> numpy array
    with wave.open(wav_path) as wav:
        wav = np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16")
        wav = wav.astype("float")
    if normalize:
        return (wav - wav.mean()) / wav.std()
    else:
        return wav


def spectrogram(wav, normalize=True):
    D = librosa.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )

    spec, phase = librosa.magphase(D)
    spec = np.log1p(spec)
    spec = torch.FloatTensor(spec)

    if normalize:
        spec = (spec - spec.mean()) / spec.std()

    return spec
class MASRDataset2(Dataset):
    def __init__(self, index_path,index_path2, labels_path):
        with open(index_path,'r',encoding='utf-8') as f:
            idx1 = f.readlines()
        idx1 = [x.strip().split(",", 1) for x in idx1]
        self.idx1 = idx1
        with open(index_path2,'r',encoding='utf-8') as f2:
            idx2 = f2.readlines()
        idx2 = [x.strip().split(",", 1) for x in idx2]
        self.idx2 = idx2
        labels = joblib.load(labels_path)
        new_word_ad = [w.strip() for w in open('./newword.csv', 'r', encoding='utf-8').readlines()]
        labels.extend(new_word_ad)
        self.labels = dict([(labels[i], i) for i in range(len(labels))])
        self.labels_str = labels

    def __getitem__(self, index):
        wav, transcript = self.idx1[index]
        wav = load_audio(wav)
        spect = spectrogram(wav)
        transcript = list(filter(None, [self.labels.get(x) for x in transcript]))

        return spect, transcript

    def __len__(self):
        return len(self.idx1)

class MASRDataset(Dataset):
    def __init__(self, index_path, labels_path):
        with open(index_path,'r',encoding='utf-8') as f:
            idx = f.readlines()
        idx = [x.strip().split(",", 1) for x in idx]
        self.idx = idx
        labels = joblib.load(labels_path)
        new_word_ad = [w.strip() for w in open('./newword.csv', 'r', encoding='utf-8').readlines()]
        labels.extend(new_word_ad)
        self.labels = dict([(labels[i], i) for i in range(len(labels))])
        self.labels_str = labels

    def __getitem__(self, index):
        wav, transcript = self.idx[index]
        wav = load_audio(wav)
        spect = spectrogram(wav)
        transcript = list(filter(None, [self.labels.get(x) for x in transcript]))

        return spect, transcript

    def __len__(self):
        return len(self.idx)


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_lens = torch.IntTensor(minibatch_size)
    target_lens = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        input_lens[x] = seq_length
        target_lens[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    # print('target.shape :',targets.shape)
    return inputs, targets, input_lens, target_lens


class MASRDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MASRDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

