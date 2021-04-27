import time
from datetime import timedelta

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

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))