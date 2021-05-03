import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from common import TRAIN, UNK, PAD, TXT_EXTENSION
from data_loader.base_loader import BaseLoader


class TextDataset(Dataset):
    def __init__(self, text_path, token, vocab_dict, seq_len=32):
        super(TextDataset, self).__init__()
        self.text_path = text_path
        self.token = token
        self.vocab_dict = vocab_dict
        self.seq_len = seq_len
        self.inputs, self.labels = self._load_dataset()
        self.classes = torch.unique(self.labels)

    def _load_dataset(self):
        inputs = []
        labels = []
        with open(self.text_path, 'r', encoding="UTF-8") as f:
            for line in tqdm(f):
                line_str = line.strip()
                if not line_str:
                    return inputs, labels
                input_, label_ = line_str.split("\t")
                words = self.token(input_)
                padding_words = self._add_pad(words)
                words_vocab = self._to_vocab(padding_words)
                inputs.append(words_vocab)
                labels.append(label_)
        inputs = torch.LongTensor(np.array(inputs).astype("int64"))
        labels = torch.LongTensor(np.array(labels).astype("int64"))
        return inputs, labels

    def _add_pad(self, words):
        if self.seq_len:
            if len(words) < self.seq_len:
                words.extend([PAD] * (self.seq_len - len(words)))
            else:
                words = words[:self.seq_len]
        return words

    def _to_vocab(self, words):
        words_vocab = []
        for word in words:
            words_vocab.append(self.vocab_dict.get(word, self.vocab_dict.get(UNK)))
        return words_vocab

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __len__(self):
        return len(self.inputs)


class TextLoader(BaseLoader):
    def __init__(self, opt):
        super(TextLoader, self).__init__(opt)
        self.use_word = opt.word
        self.vocab = opt.vocab if opt.vocab is not None and os.path.exists(opt.vocab) else None
        self.vocab_dict = self._build_vocab()
        self.seq_len = opt.seq_len
        self.vocab_size = len(self.vocab_dict)

    def tokenizer(self, words):
        if self.use_word:
            return words.split(" ")
        else:
            return [word for word in words]

    def _build_vocab(self, min_freq=1, max_size=10000):
        if self.vocab:
            vocab_dict = pickle.load(open(self.vocab, 'rb'))
        else:
            vocab_dict = {}
            self.vocab = os.path.join(self.data_dir, "vocab.pkl")
            self.train_text = os.path.join(self.data_dir, "train.txt")
            if not os.path.exists(self.train_text):
                print("no train data and no vocab data. ")
                exit(-1)
            print("Building vocab...")
            with open(self.train_text, 'r', encoding="UTF-8") as f:
                for line in tqdm(f):
                    line_str = line.strip()
                    if not line_str:
                        break
                    words = line_str.split("\t")[0]
                    words = self.tokenizer(words)
                    for word in words:
                        vocab_dict[word] = vocab_dict.get(word, 0) + 1
            vocab_list = [vocab for vocab in vocab_dict.items() if vocab[1] >= min_freq]
            vocab_list = sorted(vocab_list, key=lambda x: x[1], reverse=True)[:max_size]
            vocab_dict = {word: idx for idx, (word, count_) in enumerate(vocab_list)}
            vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict) + 1})
            pickle.dump(vocab_dict, open(self.vocab, 'wb'))
        print("Vocab Size:", len(vocab_dict))
        return vocab_dict

    def _load_sets(self, tag=TRAIN, **kwargs):
        if not os.path.exists(self.data_dir):
            print("Missing data folders")
            exit(-1)
        data_path = os.path.join(self.data_dir, tag + TXT_EXTENSION)
        if not os.path.exists(data_path):
            print("Missing {} texts. ".format(tag))
            exit(-1)
        data_sets = TextDataset(data_path, self.tokenizer, self.vocab_dict, seq_len=self.seq_len)
        return data_sets
