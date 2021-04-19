import torch.nn as nn


class LossFunc:
    def __init__(self, opt):
        self.loss_func_name = opt.loss
        loss_func = getattr(nn, self.loss_func_name)
        self.loss_func = loss_func()
        self.loss = 0.0

    def compute_(self, outputs, labels):
        self.loss = self.loss_func(outputs, labels)
        self.loss.backward()
        return self.loss
