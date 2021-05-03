import time

from common import get_time_dif, set_seed
from criterion.criterion import LossFunc
from data_loader.image_loader import ImageLoader
from model.image_classify_model import PreCNNModel
from model.save import ModelSave
from optimizer.optimizer import Optimizer
from options.image_classify_opt import ImageClassifyOpt
from plot.image_classify_plot import ImageClassifierPlot
from run_func import train, test

if __name__ == '__main__':
    opt = ImageClassifyOpt()
    opt = opt.args
    is_train = (opt.test is False)
    set_seed(opt.seed)  # 保证能复现结果

    model = PreCNNModel(opt)
    print("Loading data...")
    start_time = time.time()
    data = ImageLoader(opt)
    data.load(model.input_size)
    print("Time usage:", get_time_dif(start_time))
    model.build(data.class_nums)
    criterion = LossFunc(opt)

    if not is_train:
        print("Testing data...")
        test(model, data, criterion, opt.out)
    else:
        print("Transfer Training data...")
        saved = ModelSave(opt)
        plot = ImageClassifierPlot(opt)
        plot.write_information()
        plot.write_model(model.model, data.device, model.input_size)
        optimizer = Optimizer(opt, model.model.parameters())
        train(opt, data, model, optimizer, criterion, saved, plot)
        # 再继续学习所有参数
        if opt.again:
            print("Transfer Training data again...")
            print("Set all params requiring grad...")
            model.set_params_requires_grad(use_pre=False)
            opt.epoch_num = opt.epoch_num // 2
            train(opt, data, model, optimizer, criterion, saved, plot)
        plot.write_done()
        saved.save_model(model.model)
