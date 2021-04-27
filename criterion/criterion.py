import torch.nn as nn


class LossFunc:
    def __init__(self, opt):
        self.func_name = opt.loss
        loss_func = getattr(nn, self.func_name)
        self.loss_func = loss_func()
        self.loss = None

    def compute_(self, outputs, labels):
        self.loss = self.loss_func(outputs, labels)
        return self.loss

    def backward(self):
        self.loss.backward()
