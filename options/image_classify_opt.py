from .base_opt import BaseOpt
from common import CNN_MODELS


class ImageClassifyOpt(BaseOpt):
    def __init__(self):
        super(ImageClassifyOpt, self).__init__()

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--model', type=str, default='ResNet', choices=CNN_MODELS,
                            help='chooses which model to use. ')
        BaseOpt.add_model_args(parser)

    @staticmethod
    def add_resource_args(parser):
        parser.add_argument('--data', required=True,
                            help="for train, the path should have sub-folders train and valid;for test,"
                                 " should have sub-folders test. reference docs: "
                                 "https://pytorch.org/vision/stable/datasets.html#imagefolder")
        BaseOpt.add_resource_args(parser)


if __name__ == '__main__':
    opt = ImageClassifyOpt()
    args = opt.args
