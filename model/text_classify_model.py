import os

import numpy as np
import torch
import torch.nn as nn

from common import EMBEDDINGS_FILE
from model.base_model import BaseModel


class TextRNN(nn.Module):
    def __init__(self, work_goal, pretrained_embed=None):
        super(TextRNN, self).__init__()
        if pretrained_embed is None:
            embed_size = 0  # 词向量大小为0
            embed_dim = 300  # 词向量维度为300维
            pad_idx = -1
            self.embedding = nn.Embedding(embed_size, embed_dim, padding_idx=pad_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
            # freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
            # Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            embed_dim = pretrained_embed.size(1)  # 词向量维度从已有词向量中获取
        hidden = 128
        self.lstm = nn.LSTM(embed_dim, hidden_size=hidden, num_layers=2, bidirectional=True, batch_first=True,
                            dropout=0.5)
        self.fc = nn.Linear(hidden * 2, work_goal)  # 隐层特征向量进行拼接，所以乘以2

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)  # 最后时刻的 hidden state
        return x


class TextCNN(nn.Module):
    # todo
    def __init__(self, work_goal, pretrained_embed=None):
        super(TextCNN, self).__init__()
        if pretrained_embed is None:
            embed_size = 0  # 词向量大小为0
            embed_dim = 300  # 词向量维度为300维
            self.embedding = nn.Embedding(embed_size, embed_dim, padding_idx=embed_size - 1)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
            embed_dim = pretrained_embed.size(1)  # 词向量维度从已有词向量中获取
        filter_width = embed_dim
        filter_height = (2, 3, 4)
        filter_size = [(k, filter_width) for k in filter_height]
        filter_channels = 256
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(1, filter_channels, _size) for _size in filter_size]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(filter_channels * len(filter_height), work_goal)

    def forward(self, x):
        # todo
        return x


class TextClassifyModel(BaseModel):
    def __init__(self, opt):
        super(TextClassifyModel, self).__init__(opt)
        self.data_dir = opt.data
        self.embedding = opt.embedding
        self.pretrained_embed = None
        self._set_pretrained_embed()

    def _set_pretrained_embed(self):
        self.embed_file = EMBEDDINGS_FILE.get(self.embedding)
        if self.embed_file:
            self.embed_file = os.path.join(self.data_dir, self.embed_file)
            if not os.path.exists(self.embed_file):
                print("Missing embed files. ")
                exit(-1)
            else:
                self.pretrained_embed = np.load(self.embed_file)["embeddings"].astype('float32')
                self.pretrained_embed = torch.tensor(self.pretrained_embed)
        else:
            self.pretrained_embed = None

    def _set_model(self):
        if self.model_name == "TextRNN":
            self.model = TextRNN(self.work_goal, pretrained_embed=self.pretrained_embed)
        elif self.model_name == "TextCNN":
            self.model = TextCNN(self.work_goal, pretrained_embed=self.pretrained_embed)
