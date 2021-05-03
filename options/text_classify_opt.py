from common import TEXT_MODELS, EMBEDDINGS
from options.base_opt import BaseOpt


class TextClassifyOpt(BaseOpt):
    def __init__(self):
        super(TextClassifyOpt, self).__init__()

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--model', type=str, default='TextRNN', choices=TEXT_MODELS,
                            help='choose which model to use. ')
        parser.add_argument('--embedding', type=str, default='Sogou', choices=EMBEDDINGS,
                            help="choose embeddings.")
        parser.add_argument('--seq_len', type=int, default=32, help="fixed sentence length.")
        parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
        BaseOpt.add_model_args(parser)

    @staticmethod
    def add_resource_args(parser):
        parser.add_argument('--data', default='resources/data',
                            help="for train, the path should have train.txt and valid.txt;for test,"
                                 " should have test.txt. ")
        parser.add_argument('--out', help="out folders for test")
        parser.add_argument('--vocab', help='vocab dict path. ')
        BaseOpt.add_resource_args(parser)
