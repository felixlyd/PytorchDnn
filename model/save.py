import torch


class ModelSave:
    def __init__(self, opt_):
        self.valid_best_loss = float('inf')
        self.last_better_iter = 0
        self.path = opt_.model_save
        self.state_dict = None

    def save_model_state(self, loss_, model_, iter_):
        if loss_ < self.valid_best_loss:
            self.valid_best_loss = loss_
            self.state_dict  = model_.state_dict()
            torch.save(self.state_dict, self.path + ".state_dict")
            self.last_better_iter = iter_

    def is_shut_down(self, iter_):
        max_iter = 2000
        if iter_ - self.last_better_iter > max_iter:
            print("No optimization for a long time({} iters), auto-stopping...".format(max_iter))
            self.last_better_iter = iter_ # for second training
            return True
        else:
            return False

    def save_model(self, model_):
        model_.load_state_dict(self.state_dict)
        torch.save(model_, self.path)