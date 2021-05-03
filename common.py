import random
import time
from datetime import timedelta

import numpy as np
import torch

# py-dnn

APP_DESCRIPTION = "-" * 20 + "My Py-DNN by Pytorch" + "-" * 20 + "\n"

TRAIN = "train"
VALID = "valid"
TEST = "test"

OPTIMIZERS = [
    'Adam',
]

LOSS_FUNCTIONS = [
    'NLLLoss',
    'CrossEntropyLoss',  # CrossEntropyLoss = LogSoftMax + NLLloss
]

LR_SCHEDULERS = [
    'StepLR',
    # 'MultiStepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
]


def set_seed(seed):
    if seed is None:
        seed = 24
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# image_classify_model

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

CNN_MODELS = [
    'VGG',
    'ResNet',
    'DenseNet',
    'ResNext',
]

# text_classify_model

TEXT_MODELS = [
    'TextRNN',
    'TextCNN'
]

EMBEDDINGS = [
    'Tencent',
    'Sogou',
    'Random'
]

EMBEDDINGS_FILE = {
    'Tencent': "embedding_Tencent.npz",
    'Sogou': "embedding_SougouNews.npz",
}

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
TXT_EXTENSION = '.txt'
