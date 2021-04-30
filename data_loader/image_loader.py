import os

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from common import TRAIN, VALID, TEST

'''
使用pytorch中的torchvision.datasets模块，可以很方便的处理输入数据
官方文档：https://pytorch.org/vision/stable/datasets.html
'''


def get_transforms(is_train, input_size=224):
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomRotation(45),  # -45到45度间随机选角度旋转
                transforms.CenterCrop(input_size),  # 从中心开始裁剪到224*224
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 亮度，对比度，饱和度，色相
                transforms.RandomGrayscale(p=0.025),  # 概率转换为灰度图
                transforms.ToTensor(),  # 转换成张量Tensor格式
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 测试集输入规格保持一致
            ]
        )


def this2image(tensor):
    tensor = tensor.to("cpu").clone().detach()
    image = tensor.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


class DataIter:
    def __init__(self, dataloader, device):
        self.dataloader = [(x, y) for x, y in tqdm(dataloader)]
        self.index = 0
        self.len = len(self.dataloader)
        self.device = device

    def _to_tensor(self, x, y):
        return x.to(self.device), y.to(self.device)

    def __next__(self):
        if self.index >= self.len:
            self.index = 0
            raise StopIteration
        else:
            x, y = self.dataloader[self.index]
            self.index = self.index + 1
            x, y  = self._to_tensor(x, y)
            return x, y

    def __iter__(self):
        return self

    def __len__(self):
        return self.len


class ImageLoader():
    def __init__(self, opt):
        self.data_dir = opt.data
        self.batch_size = opt.batch_size
        self.train_loaded = None
        self.class_nums = 0
        self.valid_loaded = None
        self.test_loaded = None
        self.thread = opt.thread_num
        self.gpu_ids = opt.gpu_ids
        self.device = self._set_device()
        self.inputs = {}
        self.is_train = (opt.test is False)

    def _load_train_valid(self, input_size=224):
        if not os.path.exists(self.data_dir):
            print("Missing data folders")
            exit(-1)
        train_data_dir = os.path.join(self.data_dir, TRAIN)
        valid_data_dir = os.path.join(self.data_dir, VALID)
        if not os.path.exists(train_data_dir) or not os.path.exists(valid_data_dir):
            print("Missing {} and {} folders".format(TRAIN, VALID))
            exit(-1)
        train_transforms = get_transforms(True, input_size)
        train_sets = datasets.ImageFolder(train_data_dir, train_transforms)
        self.train_loaded = DataLoader(train_sets, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.thread)
        valid_transforms = get_transforms(False, input_size)
        valid_sets = datasets.ImageFolder(valid_data_dir, valid_transforms)
        self.valid_loaded = DataLoader(valid_sets, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.thread)
        self.class_nums = len(train_sets.classes)
        print("Train sets:", len(train_sets))
        print("Valid sets:", len(valid_sets))
        print("Batch size:", self.batch_size)
        print("Class nums:", self.class_nums)

    def _load_test(self, input_size=224):
        if not os.path.exists(self.data_dir):
            print("Missing data folders")
            exit(-1)
        test_data_dir = os.path.join(self.data_dir, TEST)
        if not os.path.exists(test_data_dir):
            print("Missing {} folders".format(TEST))
            exit(-1)
        test_sets = datasets.ImageFolder(test_data_dir, get_transforms(False, input_size))
        self.test_loaded = DataLoader(test_sets, batch_size=self.batch_size, num_workers=self.thread)

    def init_(self, input_size=224):
        if self.is_train:
            self._load_train_valid(input_size)
            self.inputs[TRAIN] = DataIter(self.train_loaded, self.device)
            self.inputs[VALID] = DataIter(self.valid_loaded, self.device)
        else:
            self._load_test(input_size)
            self.inputs[TEST] = DataIter(self.test_loaded, self.device)

    def _set_device(self):
        device = "cpu"
        cuda = torch.cuda.is_available()
        if self.gpu_ids[0] != -1:
            if cuda:
                device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
            else:
                device = torch.device('cpu')
        return device
