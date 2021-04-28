import random
import time
from datetime import timedelta

import numpy as np
import torch

APP_DESCRIPTION = "-" * 20 + "My Py-DNN by Pytorch" + "-" * 20 + "\n"

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

TRAIN = "train"
VALID = "valid"
TEST = "test"

CNN_MODELS = [
    'VGG',
    'ResNet',
    'DenseNet',
    'ResNext',
]

OPTIMIZERS = [
    'Adam',
]

LOSS_FUNCTIONS = [
    'NLLLoss',
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
