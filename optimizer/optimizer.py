import torch.optim as optim


class Optimizer:
    def __init__(self, opt):
        self._optimizer = None
        self._optimizer_name = opt.optimizer
        self._lr = opt.learning_rate
        self._lr_scheduler = None
        self._lr_scheduler_name = opt.lr_scheduler
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.gamma = opt.gamma

    def _init_optimizer(self, params):
        # todo
        if self._optimizer_name == "Adam":
            self._optimizer = optim.Adam(params, lr=self._lr, betas=(self.beta1,self.beta2))

    def _init_lr_scheduler(self):
        # todo
        if self._lr_scheduler_name is None:
            self._lr_scheduler = None
        elif self._lr_scheduler_name == "StepLR":
            self._lr_scheduler = optim.lr_scheduler.StepLR(self._optimizer, step_size=7, gamma=self.gamma)
