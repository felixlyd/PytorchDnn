from data_loader.image_loader import ImageLoader

DATA_OPT = {
    'image': ImageLoader,
}


def set_data(model_type):
    return DATA_OPT[model_type]
