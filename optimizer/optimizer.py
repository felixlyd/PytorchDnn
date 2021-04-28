import torch.optim as optim


class Optimizer:
    def __init__(self, opt, params):
        self._optimizer = None
        self._optimizer_name = opt.optimizer
        self._lr = opt.learning_rate
        self._lr_scheduler = None
        self._lr_scheduler_name = opt.lr_scheduler
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.gamma = opt.gamma

        self._init_optimizer(params)
        self._init_lr_scheduler()

    def _init_optimizer(self, params):
        # todo
        if self._optimizer_name == "Adam":
            self._optimizer = optim.Adam(params, lr=self._lr, betas=(self.beta1,self.beta2))

    def _init_lr_scheduler(self):
        if self._lr_scheduler_name is None:
            self._lr_scheduler = None
        elif self._lr_scheduler_name == "StepLR":
            self._lr_scheduler = optim.lr_scheduler.StepLR(self._optimizer, step_size=7, gamma=self.gamma)
        # elif self._lr_scheduler_name == "MultiStepLR":
        #     pass
        elif self._lr_scheduler_name == "ExponentialLR":
            self._lr_scheduler = optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=self.gamma)
        elif self._lr_scheduler_name == "CosineAnnealingLR":
            self._lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=10)

    def zero_(self):
        self._optimizer.zero_grad()

    def update_(self):
        self._optimizer.step()

    def lr_decay(self):
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def get_iter_lr(self):
        return self._optimizer.param_groups[0]['lr']
