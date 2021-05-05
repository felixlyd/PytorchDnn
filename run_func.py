import time

from common import TRAIN, VALID, get_time_dif, TEST
from evaluate.image_classify_eval import get_train_acc, get_test_acc_loss


def train(opt, data, model, optimizer, criterion, saved, plot):
    start_time = time.time()
    model.model.train()
    for epoch in range(opt.epoch_num):
        print('Epoch [{}/{}]'.format(epoch + 1, opt.epoch_num))
        for i, (inputs, labels) in enumerate(data.inputs[TRAIN]):
            optimizer.zero_()
            outputs = model.model(inputs)
            train_loss = criterion.compute_(outputs, labels)
            criterion.backward()
            optimizer.update_()
            if model.total_iter % opt.plot_freq == 0:
                train_acc = get_train_acc(outputs, labels)
                valid_acc, valid_loss = get_test_acc_loss(model.model, data.inputs[VALID], criterion)
                model.model.train()
                saved.save_model_state(valid_loss, model.model, model.total_iter)
                plot.write_loss_acc(train_loss.item(), valid_loss, train_acc, valid_acc, model.total_iter)
                plot.print_msg(train_loss.item(), valid_loss, train_acc, valid_acc, model.total_iter,
                               get_time_dif(start_time))
                plot.write_lr(optimizer.get_iter_lr(), model.total_iter)
            model.total_iter = model.total_iter + 1
            if saved.is_shut_down(model.total_iter):
                return
        optimizer.lr_decay()
    print("Time usage:", get_time_dif(start_time))


def test(model, data, criterion, out_path):
    start_time = time.time()
    test_acc, test_loss = get_test_acc_loss(model.model, data.inputs[TEST], criterion,test=True, out_path=out_path)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Time usage:", get_time_dif(start_time))
