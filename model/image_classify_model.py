import torch
from torchvision import models
import torch.nn as nn

'''
这里的迁移学习主要参考官方文档：https://pytorch.org/vision/stable/models.html
要注意模型的输入size对应，之后将模型的全连接层按自己所需修改
'''

class PreCNNModel:
    def __init__(self, opt):
        self.model = None
        self.model_name = opt.model
        self.work_goal = 0
        self.input_size = 0
        self._set_input_size()
        self.gpu_ids = opt.gpu_ids
        self.model_save = opt.model_save
        self.device = None
        self.epochs = opt.epoch_num
        self.total_iter = 0
        self.is_train = (opt.test is False)

    # Transfer Learning
    def _set_params_requires_grad(self, use_pre=True):
        if use_pre:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def _set_model(self, use_pre=True):
        if self.model_name == "VGG":
            self.model = models.vgg16(pretrained=use_pre)
            self._set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )
        elif self.model_name == "ResNet":
            self.model = models.resnet152(pretrained=use_pre)
            self._set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )
        elif self.model_name == "DenseNet":
            self.model = models.densenet169(pretrained=use_pre)
            self._set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )
        elif self.model_name == "ResNext":
            self.model = models.resnext50_32x4d(pretrained=use_pre)
            self._set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )

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

    def _set_input_size(self):
        if self.model_name == "VGG":
            self.input_size = 224
        elif self.model_name == "ResNet":
            self.input_size = 224
        elif self.model_name == "DenseNet":
            self.input_size = 224
        elif self.model_name == "ResNext":
            self.input_size = 224

    def init_(self, work_goal=0):
        if self.is_train:
            self.work_goal = work_goal
            self._set_model()
            self._set_device()
        else:
            self.model = torch.load(self.model_save)
            self._set_device()
