from torchvision import models
import torch.nn as nn

from model.base_model import BaseModel

'''
这里的迁移学习主要参考官方文档：https://pytorch.org/vision/stable/models.html
要注意模型的输入size对应，之后将模型的全连接层按自己所需修改
'''


class PreCNNModel(BaseModel):
    def __init__(self, opt):
        super(PreCNNModel, self).__init__(opt)
        self._set_input_size()

    # Transfer Learning
    def set_params_requires_grad(self, use_pre=True):
        if use_pre:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def _set_input_size(self):
        if self.model_name == "VGG":
            self.input_size = 224
        elif self.model_name == "ResNet":
            self.input_size = 224
        elif self.model_name == "DenseNet":
            self.input_size = 224
        elif self.model_name == "ResNext":
            self.input_size = 224

    def _set_model(self, use_pre=True):
        if self.model_name == "VGG":
            self.model = models.vgg16(pretrained=use_pre)
            self.set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )
        elif self.model_name == "ResNet":
            self.model = models.resnet152(pretrained=use_pre)
            self.set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )
        elif self.model_name == "DenseNet":
            self.model = models.densenet169(pretrained=use_pre)
            self.set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )
        elif self.model_name == "ResNext":
            self.model = models.resnext50_32x4d(pretrained=use_pre)
            self.set_params_requires_grad(use_pre=use_pre)
            feature_nums = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(feature_nums, self.work_goal),
                nn.LogSoftmax(dim=1),
            )
