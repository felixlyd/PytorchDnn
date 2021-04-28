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


def transfer_train(opt_, data_, model_, optimizer_, criterion_, saved_, plot_):
    model_.model.train()
    total_iter = 0
    stop = False
    for epoch in range(opt_.epoch_num):
        print('Epoch [{}/{}]'.format(epoch + 1, opt.epoch_num))
        for i, (inputs, labels) in enumerate(data_.inputs[TRAIN]):
            optimizer_.prepare()
            inputs = inputs.to(data.device)
            labels = labels.to(data.device)
            outputs = model_.model(inputs)
            train_loss = criterion_.compute_(outputs, labels)
            criterion_.backward()
            optimizer_.update_()
            if total_iter % 100 == 0:
                train_acc = get_train_acc(outputs, labels)
                valid_acc, valid_loss = get_valid_acc_loss(model.model, data.inputs[VALID], criterion, data.device)
                model_.model.train()
                saved_.save_model_state(valid_loss, model.model, total_iter)
                plot_.write_loss_acc(train_loss.item(), valid_loss, train_acc, valid_acc, total_iter)
                plot_.print_msg(train_loss.item(), valid_loss, train_acc, valid_acc, total_iter,
                                get_time_dif(start_time))
                plot_.write_lr(optimizer.get_iter_lr(), total_iter)
            total_iter = total_iter + 1
            stop = saved.is_shut_down(total_iter)
            if stop:
                break
        optimizer_.lr_decay()
        if stop:
            break
    plot_.write_done()
    saved_.save_model(model.model)


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
    criterion = LossFunc(opt)

    if not is_train:
        start_time = time.time()
        print("Testing data...")
        test_acc, test_loss = get_test_acc_loss(model.model, data.inputs[TEST], criterion, data.device,
                                                out_path=opt.out)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Time usage:", get_time_dif(start_time))
    else:
        print("Transfer Training data...")
        start_time = time.time()
        saved = ModelSave(opt)
        plot = PlotImageClassifier(opt)
        plot.write_information()
        plot.write_model(model.model, model.input_size, data.device)
        optimizer = Optimizer(opt, model.model.parameters())
        transfer_train(opt, data, model, optimizer, criterion, saved, plot)
        print("Time usage:", get_time_dif(start_time))
        # 再继续学习所有参数
        if opt.again:
            model._set_params_requires_grad(use_pre=False)
            opt.epoch_num=10
            print("Transfer Training data again...")
            print("Set all params requiring grad...")
            start_time = time.time()
            transfer_train(opt, data, model, optimizer, criterion, saved, plot)
            print("Time usage:", get_time_dif(start_time))
