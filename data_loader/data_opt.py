import time
from datetime import timedelta

from data_loader.image_loader import ImageLoader

DATA_OPT = {
    'image': ImageLoader,
}


def set_data(model_type):
    return DATA_OPT[model_type]

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


