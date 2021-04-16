from torchvision import models
import torch.nn as nn

from common import TRAIN


class PreCNNModel:
    def __init__(self, opt):
        self.model = None
        self.model_name = opt.model
        self.work_goal = 1
        self.input_size = 0
        self.is_train = (opt.work == TRAIN)
        self.gpu_ids = opt.set_gpu()
        self.model_save = opt.model_save
        self.optimizers = None
        self.loss = None
        self.input_data = None
        self.epochs = opt.epoch_num

    # Transfer Learning
    def set_params_requires_grad(self, use_pre=True):
        if use_pre:
            for param in self.model.parameters():
                param.requires_grad = False

    def set_work_goal(self, goals):
        self.work_goal = goals

    def set_model(self, use_pre=True):
        # todo
        if self.model_name == "VGG":
            self.model = models.vgg16(pretrained=use_pre)
            self.set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(feature_nums, self.work_goal)
            self.input_size = 224
        elif self.model_name == "ResNet":
            self.model = models.resnet152(pretrained=use_pre)
            self.set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )
            self.input_size = 224

    def set_input(self, input):



