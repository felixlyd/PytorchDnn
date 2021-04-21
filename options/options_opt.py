import sys
from options.image_classify_opt import ImageClassifyOpt

OPTIONS_OPT = {
    'image': ImageClassifyOpt,
}


def set_opt():
    args = sys.argv
    if len(args) < 2:
        print('try "python run.py [TYPE] -h". \n[TYPE]: image')
        exit(-1)
    model_type = args[1]
    if model_type not in OPTIONS_OPT:
        print("cannot find the model type.")
        exit(-1)
    sys.argv.pop(1)
    opts = OPTIONS_OPT[model_type](model_type)
    args = opts.args
    return args, model_type
