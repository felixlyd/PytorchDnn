import time

from tensorboardX import SummaryWriter


class PlotImageClassifier:
    def __init__(self, opt):
        self.plot = opt.plot
        self.log = opt.log
        self.msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {' \
                   '4:>6.2%},  Time: {5}'
        if self.plot:
            self.writer = SummaryWriter(log_dir=self.log + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        else:
            self.writer = None

    def writing(self, train_loss, valid_loss, train_acc, valid_acc, iter_):
        if self.plot:
            self.writer.add_scalar("loss/train", train_loss, iter_)
            self.writer.add_scalar("loss/valid", valid_loss, iter_)
            self.writer.add_scalar("acc/train", train_acc, iter_)
            self.writer.add_scalar("acc/valid", valid_acc, iter_)
        else:
            pass

    def write_down(self):
        if self.plot:
            self.writer.close()

    def print_msg(self, train_loss, valid_loss, train_acc, valid_acc, iter_, time_dif):
        print(self.msg.format(iter_, train_loss, train_acc, valid_loss, valid_acc, time_dif))