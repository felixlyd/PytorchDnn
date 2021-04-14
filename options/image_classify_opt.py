from base_opt import BaseOpt


class ImageClassifyOpt(BaseOpt):
    def __init__(self):
        super(ImageClassifyOpt, self).__init__()

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--model', type=str, default='resnet', choices=['VGG', 'ResNet', 'DenseNet', 'ResNext'],
                            help='chooses which model to use. ')
        BaseOpt.add_model_args(parser)

    @staticmethod
    def add_resource_args(parser):
        parser.add_argument('--data', required=True,
                            help="path to the images (should have sub-folders train and valid). docs: "
                                 "https://pytorch.org/vision/stable/datasets.html#imagefolder")
        BaseOpt.add_resource_args(parser)


if __name__ == '__main__':
    opt = ImageClassifyOpt()
    args = opt.args
