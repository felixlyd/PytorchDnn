import time

import torch
from common import TRAIN, TEST, VALID
from criterion.criterion import LossFunc
from data_loader.data_opt import set_data, get_time_dif
from evaluate.evaluate import EVALUATE_OPT, TRAIN_OPT
from model.models_opt import set_model
from optimizer.optimizer import Optimizer
from options.options_opt import set_opt
from plot.plot_writer import PlotImageClassifier


class ModelSave:
    def __init__(self, opt_):
        self.valid_best_loss = float('inf')
        self.last_better_iter = 0
        self.path = opt_.model_save

    def save_model_(self, loss_, model_, iter_):
        if loss_ < self.valid_best_loss:
            self.valid_best_loss = loss_
            torch.save(model_.state_dict(), self.path)
            self.last_better_iter = iter_

    def is_shut_down(self, iter_):
        if iter_ - self.last_better_iter > 1000:
            print("No optimization for a long time({} batches), auto-stopping...".format(1000))
            return True
        else:
            return False


if __name__ == '__main__':
    opt, model_type = set_opt()
    is_train = (opt.work == TRAIN)

    print("Loading data...")
    start_time = time.time()
    data = set_data(model_type)(opt)
    data.init_(is_train)
    print("Time usage:", get_time_dif(start_time))
    work_goals = data.class_nums
    model = set_model(model_type)(opt)
    model.set_work_goal(work_goals)
    model.init_()
    start_time = time.time()
    if not is_train:
        # todo
        print("Testing data...")
        model.model.eval()
        for inputs, labels in data.inputs[TEST]:
            outputs = model.model(inputs)
        print("Time usage:", get_time_dif(start_time))
    else:
        print("Training data...")
        model.model.train()
        total_iter = 0
        stop = False
        saved = ModelSave(opt)
        plot = PlotImageClassifier(opt)
        criterion = LossFunc(opt)
        optimizer = Optimizer(opt, model.model.parameters())
        for epoch in range(opt.epoch_num):
            print('Epoch [{}/{}]'.format(epoch + 1, opt.epoch_num))
            for i, (inputs, labels) in enumerate(data.inputs[TRAIN]):
                optimizer.prepare()
                outputs = model.model(inputs)
                train_loss = criterion.compute_(outputs, labels)
                criterion.backward()
                optimizer.update_()
                if total_iter % 100 == 0:
                    train_acc = TRAIN_OPT[model_type](outputs, labels)
                    valid_acc, valid_loss = EVALUATE_OPT[model_type]( model.model, data.inputs[VALID], criterion)
                    model.model.train()
                    saved.save_model_(valid_loss, model.model, total_iter)
                    plot.writing(train_loss.item(), valid_loss, train_acc, valid_acc, total_iter)
                    plot.print_msg(train_loss.item(), valid_loss, train_acc, valid_acc, total_iter,
                                   get_time_dif(start_time))
                total_iter = total_iter + 1
                # stop = saved.is_shut_down(total_iter)
                if stop:
                    break
            optimizer.lr_decay()
            if stop:
                break
        plot.write_down()
        print("Time usage:", get_time_dif(start_time))
