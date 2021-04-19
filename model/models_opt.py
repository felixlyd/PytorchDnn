from model.image_classify_model import PreCNNModel

MODELS_OPT = {
    'image': PreCNNModel,
}


def set_model(model_type):
    return MODELS_OPT[model_type]
