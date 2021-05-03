import torch
import torch.nn as nn


class BaseModel:
    def __init__(self, opt):
        self.model = None
        self.model_name = opt.model
        self.work_goal = 0
        self.gpu_ids = opt.gpu_ids
        self.model_save = opt.model_save
        self.device = None
        self.is_train = (opt.test is False)
        self.total_iter = 0 # 记录最终迭代次数

    def _set_model(self):
        pass

    def _set_device(self):
        cuda = torch.cuda.is_available()
        if len(self.gpu_ids) == 1:
            gpu_id = self.gpu_ids[0]
            if gpu_id != -1:
                if cuda:
                    print("using GPU {}.".format(self.gpu_ids[0]))
                    self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
                else:
                    print("using CPU.")
                    self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
        else:
            print("using parallel GPUs." + ",".join(self.gpu_ids))
            self.model = nn.DataParallel(self.model, self.gpu_ids, self.gpu_ids[0])

    def build(self, work_goal=0):
        if self.is_train:
            self.work_goal = work_goal
            self._set_model()
            self._set_device()
        else:
            self.model = torch.load(self.model_save)
            self._set_device()
