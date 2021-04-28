import time

from common import TRAIN, TEST, VALID, get_time_dif
from criterion.criterion import LossFunc
from data_loader.image_loader import ImageLoader
from evaluate.image_classify_eval import get_valid_acc_loss, get_train_acc, get_test_acc_loss
from model.image_classify_model import PreCNNModel
from model.save import ModelSave
from optimizer.optimizer import Optimizer
from options.image_classify_opt import ImageClassifyOpt
from plot.plot_writer import PlotImageClassifier

if __name__ == '__main__':
    opt = ImageClassifyOpt()
    opt = opt.args
    is_train = (opt.test is False)

    model = PreCNNModel(opt)
    print("Loading data...")
    start_time = time.time()
    data = ImageLoader(opt)
    data.init_(model.input_size)
    print("Time usage:", get_time_dif(start_time))
    model.init_(data.class_nums)
    start_time = time.time()
    criterion = LossFunc(opt)
    if not is_train:
        print("Testing data...")
        test_acc, test_loss = get_test_acc_loss(model.model, data.inputs[TEST], criterion, data.device, out_path=opt.out)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Time usage:", get_time_dif(start_time))
    else:
        print("Training data...")
        model.model.train()
        total_iter = 0
        stop = False
        saved = ModelSave(opt)
        plot = PlotImageClassifier(opt)
        optimizer = Optimizer(opt, model.model.parameters())
        for epoch in range(opt.epoch_num):
            print('Epoch [{}/{}]'.format(epoch + 1, opt.epoch_num))
            for i, (inputs, labels) in enumerate(data.inputs[TRAIN]):
                optimizer.prepare()
                inputs = inputs.to(data.device)
                labels = labels.to(data.device)
                outputs = model.model(inputs)
                train_loss = criterion.compute_(outputs, labels)
                criterion.backward()
                optimizer.update_()
                if total_iter % 100 == 0:
                    train_acc = get_train_acc(outputs, labels)
                    valid_acc, valid_loss = get_valid_acc_loss(model.model, data.inputs[VALID], criterion, data.device)
                    model.model.train()
                    saved.save_model_state(valid_loss, model.model, total_iter)
                    plot.writing(train_loss.item(), valid_loss, train_acc, valid_acc, total_iter)
                    plot.print_msg(train_loss.item(), valid_loss, train_acc, valid_acc, total_iter,
                                   get_time_dif(start_time))
                total_iter = total_iter + 1
                stop = saved.is_shut_down(total_iter)
                if stop:
                    break
            optimizer.lr_decay()
            if stop:
                break
        plot.write_down()
        saved.save_model(model.model)
        print("Time usage:", get_time_dif(start_time))
