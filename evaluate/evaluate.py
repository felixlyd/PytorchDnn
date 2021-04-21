import numpy as np
import torch
from sklearn import metrics


def evaluate_image_classify(model_, data_, criterion_, test=False):
    model_.eval()
    total_loss = 0.0
    acc_nums = 0
    data_nums =0
    with torch.no_grad():
        for inputs, labels in data_:
            outputs = model_(inputs)
            loss = criterion_.compute_(outputs, labels)
            total_loss = total_loss + loss
            true = labels.data
            predict = torch.max(outputs.data, 1)[1]
            acc_nums = acc_nums + torch.sum(true == predict)
            data_nums = data_nums + len(true)
    return acc_nums / data_nums, total_loss / len(data_)


def train_image_classify(outputs, labels):
    acc_nums = 0
    predict = torch.max(outputs.data, 1)[1]
    true = labels.data
    acc_nums = acc_nums + torch.sum(true == predict)
    return acc_nums / len(outputs)


EVALUATE_OPT = {
    'image': evaluate_image_classify,
}

TRAIN_OPT = {
    'image': train_image_classify,
}
