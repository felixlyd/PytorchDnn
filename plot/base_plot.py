import os
import json
import time

from tensorboardX import SummaryWriter


class BasePlot:
    def __init__(self, opt):
        self.opt = opt
        self.model_name = opt.model
        self.lr = opt.learning_rate
        self.batch_size = opt.batch_size
        self.plot = opt.plot
        self.log = opt.log
        self.msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {' \
                   '4:>6.2%},  Time: {5}'
        if self.plot:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.log, self.model_name + "_" + time.strftime('%m-%d_%H.%M', time.localtime())))
        else:
            self.writer = None

    def write_loss_acc(self, train_loss, valid_loss, train_acc, valid_acc, iter_):
        if self.plot:
            self.writer.add_scalar("loss/train", train_loss, iter_)
            self.writer.add_scalar("loss/valid", valid_loss, iter_)
            self.writer.add_scalar("acc/train", train_acc, iter_)
            self.writer.add_scalar("acc/valid", valid_acc, iter_)
        else:
            pass

    def write_done(self):
        if self.plot:
            self.writer.close()

    def print_msg(self, train_loss, valid_loss, train_acc, valid_acc, iter_, time_dif):
        print(self.msg.format(iter_, train_loss, train_acc, valid_loss, valid_acc, time_dif))

    def write_model(self, **kwargs):
        pass

    def write_weight(self, name, param):
        if self.plot:
            self.writer.add_histogram(name, param, 0)

    def write_params(self, model_):
        if self.plot:
            params = model_.named_parameters()
            for name, param in params:
                self.write_weight(name, param)

    def write_information(self):
        if self.plot:
            information = json.dumps(vars(self.opt))
            self.writer.add_text("My-PyDNN/Params", information)

    def write_pr_curve(self, labels, predicts):
        if self.plot:
            self.writer.add_pr_curve("pr_curve", labels, predicts, 0)

    def write_lr(self, lr, iter_):
        self.writer.add_scalar("train/learning_rate", lr, iter_)
