import os

import numpy as np
from torchvision import transforms, datasets
from common import TRAIN
from data_loader.base_loader import BaseLoader

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


class ImageLoader(BaseLoader):
    def __init__(self, opt):
        super(ImageLoader, self).__init__(opt)

    def _load_sets(self, tag=TRAIN, **kwargs):
        if kwargs.get("input_size") is None:
            print("Missing data input size. ")
            exit(-1)
        else:
            input_size = kwargs.get("input_size")
            if not os.path.exists(self.data_dir):
                print("Missing data folders")
                exit(-1)
            data_dir = os.path.join(self.data_dir, tag)
            if not os.path.exists(data_dir):
                print("Missing {} folders".format(tag))
                exit(-1)
            data_transforms = get_transforms(tag == TRAIN, input_size)
            data_sets = datasets.ImageFolder(data_dir, data_transforms)
            return data_sets

    def init_(self, input_size=224):
        BaseLoader.init_(self, input_size=input_size)
