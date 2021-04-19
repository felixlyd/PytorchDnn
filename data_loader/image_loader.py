import os

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from common import TRAIN, VALID, TEST

'''
使用pytorch中的torchvision.datasets模块，可以很方便的处理输入数据
官方文档：https://pytorch.org/vision/stable/datasets.html
'''

train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(45),  # -45到45度间随机选角度旋转
        transforms.CenterCrop(224),  # 从中心开始裁剪到224*224
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 亮度，对比度，饱和度，色相
        transforms.RandomGrayscale(p=0.025),  # 概率转换为灰度图
        transforms.ToTensor(),  # 转换成张量Tensor格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化
    ]
)

valid_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 测试集输入规格保持一致
    ]
)


def this_to_image(tensor):
    tensor = tensor.to("cpu").clone().detach()
    image = tensor.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


class ImageLoader():
    def __init__(self, opt):
        self.test_loaded = None
        self.test_sets = None
        self.data_dir = opt.data
        self.batch_size = opt.batch_size
        self.train_sets = None
        self.train_loaded = None
        self.class_nums = 0
        self.valid_sets = None
        self.valid_loaded = None
        self.gpu_ids = opt.gpu_ids
        self.device = None
        self._set_device()
        self.inputs = {}

    def _load_train_valid(self):
        train_data_dir = os.path.join(self.data_dir, TRAIN)
        valid_data_dir = os.path.join(self.data_dir, VALID)
        if not os.path.exists(train_data_dir) or not os.path.exists(valid_data_dir):
            print("Missing {} and {} folders".format(TRAIN, VALID))
            exit(-1)
        self.train_sets = datasets.ImageFolder(train_data_dir, train_transforms)
        self.train_loaded = DataLoader(self.train_sets, batch_size=self.batch_size, shuffle=True)
        self.valid_sets = datasets.ImageFolder(valid_data_dir, valid_transforms)
        self.valid_loaded = DataLoader(self.valid_sets, batch_size=self.batch_size, shuffle=True)
        self.class_nums = len(self.train_sets.classes)

    def _load_test(self):
        # todo
        test_data_dir = os.path.join(self.data_dir, TEST)
        if not os.path.exists(test_data_dir):
            print("Missing {} folders".format(TEST))
            exit(-1)
        self.test_sets = datasets.ImageFolder(test_data_dir, train_transforms)
        self.test_loaded = DataLoader(self.test_sets, batch_size=self.batch_size, shuffle=True)

    def init_(self, is_train):
        if is_train:
            self._load_train_valid()
            self.inputs[TRAIN] = self.train_loaded.to(self.device)
            self.inputs[VALID] = self.valid_loaded.to(self.device)
        else:
            self._load_test()
            self.inputs[TEST] = self.test_loaded.to(self.device)

    def _set_device(self):
        cuda = torch.cuda.is_available()
        if self.gpu_ids[0] != -1:
            if cuda:
                self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
            else:
                self.device = torch.device('cpu')