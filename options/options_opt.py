import sys
from options.image_classify_opt import ImageClassifyOpt

OPTIONS_OPT = {
    'image': ImageClassifyOpt,
}


def set_opt():
    args = sys.argv
    model_type = args[1]
    if model_type not in OPTIONS_OPT:
        print("cannot find the model type.")
        exit(-1)
    sys.argv.pop(1)
    opts = OPTIONS_OPT[model_type]()
    args = opts.args
    return args, model_type
