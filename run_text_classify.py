import time

from common import set_seed, get_time_dif
from criterion.criterion import LossFunc
from data_loader.text_loader import TextLoader
from model.save import ModelSave
from model.text_classify_model import TextClassifyModel
from optimizer.optimizer import Optimizer
from options.text_classify_opt import TextClassifyOpt
from plot.text_classify_plot import TextClassifierPlot
from run_func import test, train

if __name__ == '__main__':
    opt = TextClassifyOpt()
    opt = opt.args
    is_train = (opt.test is False)
    set_seed(opt.seed)

    model = TextClassifyModel(opt)
    print("Loading data...")
    start_time = time.time()
    data = TextLoader(opt)
    data.load()
    print("Time usage:", get_time_dif(start_time))
    model.build(data.class_nums)
    criterion = LossFunc(opt)

    if not is_train:
        print("Testing data...")
        test(model, data, criterion, opt.out)
    else:
        print("Transfer Training data...")
        saved = ModelSave(opt)
        plot = TextClassifierPlot(opt)
        plot.write_information()
        optimizer = Optimizer(opt, model.model.parameters())
        train(opt, data, model, optimizer, criterion, saved, plot)
        plot.write_done()
        saved.save_model(model.model)
