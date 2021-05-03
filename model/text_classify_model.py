import os

import numpy as np
import torch
import torch.nn as nn

from common import EMBEDDINGS_FILE
from model.base_model import BaseModel


class TextRNN(nn.Module):
    def __init__(self, work_goal, pretrained_embed=None, embed_size=0):
        super(TextRNN, self).__init__()
        if pretrained_embed is None:
            embed_dim = 300  # 词向量维度为300维
            self.embedding = nn.Embedding(embed_size, embed_dim, padding_idx=embed_size - 1)
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
        # x.size [batch_size, seq_len]
        x = self.embedding(x)
        # x.size [batch_size, seq_len, embedding]
        x, _ = self.lstm(x)
        # x.size [batch_size,  seq_len, hidden_size * 2]
        x = x[:, -1, :]
        # 只取seq中的最后一个，代表最后时刻 x,size [batch_size, hidden_size * 2]
        x = self.fc(x)
        # x.size [batch_size, work_goal]
        return x


class TextCNN(nn.Module):
    def __init__(self, work_goal, seq_len, pretrained_embed=None, embed_size=0):
        super(TextCNN, self).__init__()
        if pretrained_embed is None:
            embed_dim = 300  # 词向量维度为300维
            self.embedding = nn.Embedding(embed_size, embed_dim, padding_idx=embed_size - 1)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
            embed_dim = pretrained_embed.size(1)  # 词向量维度从已有词向量中获取
        # x.size [batch_size, seq_len, embedding]
        window_sizes = (2, 3, 4)  # 每h个词成为一个窗口
        filter_channels = 256
        self.conv_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=filter_channels, kernel_size=(h,)),
                # x.size [batch_size, filter_channels, seq_len - h + 1]
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=seq_len - h + 1)
                # x.size [batch_size, filter_channels, 1]
            ) for h in window_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(filter_channels * len(window_sizes), work_goal)

    def forward(self, x):
        # x.size [batch_size, seq_len]
        x = self.embedding(x)
        # x.size [batch_size, seq_len, embedding]
        x = x.permute(0, 2, 1)
        # x.size [batch_size, embedding, seq_len]
        x = [conv(x) for conv in self.conv_list]
        # x: a list x[i].size [batch_size, filter_channels, 1]
        x = torch.cat(x, dim=1)
        # x.size [batch_size, filter_channels * len(windows_sizes), 1]
        x = x.squeeze(dim=2)
        # x.size [batch_size, filter_channels * len(windows_sizes)]
        x = self.fc(x)
        # x.size [batch_size, work_goal]
        return x


class TextClassifyModel(BaseModel):
    def __init__(self, opt):
        super(TextClassifyModel, self).__init__(opt)
        self.data_dir = opt.data
        self.embedding = opt.embedding
        self.pretrained_embed = None
        self._set_pretrained_embed()
        self.seq_len = opt.seq_len
        self.embed_size = 0  # 词向量数目，如果预词向量为空，需要自己生成词向量

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
            self.model = TextRNN(self.work_goal, pretrained_embed=self.pretrained_embed, embed_size=self.embed_size)
        elif self.model_name == "TextCNN":
            self.model = TextCNN(self.work_goal, self.seq_len, pretrained_embed=self.pretrained_embed,
                                 embed_size=self.embed_size)

    def build(self, work_goal=0, embed_size=0):
        if self.pretrained_embed is None and self.is_train:
            self.embed_size = embed_size
            self.work_goal = work_goal
            self._set_model()
            self._set_device()
        else:
            BaseModel.build(self, work_goal=work_goal)
