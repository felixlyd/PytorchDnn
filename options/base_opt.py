import argparse

from common import APP_DESCRIPTION, TRAIN, TEST, OPTIMIZERS, LOSS_FUNCTIONS, LR_SCHEDULERS


class BaseOpt:
    def __init__(self):
        self.parser = None
        self.args = None
        self.init_state = False
        if not self.init_state:
            self._init()
            self.init_state = True
        self.args.gpu_ids = self._set_gpu()
        self.print_opt()

    def _init(self):
        parser = argparse.ArgumentParser(
            description=APP_DESCRIPTION,
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.add_args(parser)
        self.parser = parser
        self.args = parser.parse_args()
        if self.args.help:
            self.parser.print_help()
            exit(0)

    def add_args(self, parser):
        resource_parser = parser.add_argument_group("Resource Arguments")
        model_parser = parser.add_argument_group("Model Arguments")
        optimizer_parser = parser.add_argument_group("Optimizer Arguments")
        other_parser = parser.add_argument_group("Other Arguments")
        self.add_resource_args(resource_parser)
        self.add_model_args(model_parser)
        self.add_optimizer_args(optimizer_parser)
        self.add_other_args(other_parser)

    @staticmethod
    def add_resource_args(parser):
        parser.add_argument('--log', default="resources/log", help="path to the log folder to record information.")
        parser.add_argument('--model_save', default="resources/saved_model/model.pth", help="models are saved here.")

    @staticmethod
    def add_model_args(parser):
        parser.add_argument('--test', action="store_true", help="chooses model work type to test.")
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size.')
        parser.add_argument('--epoch_num', type=int, default=20, help='epoch size.')

    @staticmethod
    def add_optimizer_args(parser):
        parser.add_argument('--optimizer', type=str, default='Adam', choices=OPTIMIZERS,
                            help="chooses which optimizer to use. ")
        parser.add_argument('--learning_rate', type=float, default=0.001, help="initial learning rate.")
        parser.add_argument('--beta1', type=float, default=0.9, help="possible parameters named by beta1.")
        parser.add_argument('--beta2', type=float, default=0.999, help="possible parameters named by beta1.")
        parser.add_argument('--lr_scheduler', type=str, choices=LR_SCHEDULERS, help="chooses which lr_scheduler to use.")
        parser.add_argument('--gamma', type=float, default=0.1, help="gamma parameter of lr_scheduler.")

    @staticmethod
    def add_other_args(parser):
        parser.add_argument('--loss', type=str, default='NLLLoss', choices=LOSS_FUNCTIONS,
                            help="chooses which loss function to use.")
        parser.add_argument('--thread_num', type=int, default=4, help="threads for loading data.")
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU.')
        parser.add_argument('--plot', action="store_true", help='if specified, plot the logs powered by tensorboard.')
        parser.add_argument('--plot_freq', type=int, default=100, help='every X iters to record loss and acc. ')
        parser.add_argument('--seed', type=int, default=24, help="random seed.")
        parser.add_argument('--help', action="store_true", help="show this help message and exit.")

    def print_opt(self):
        if self.args.help:
            self.parser.print_help()
            exit()
        message = ""
        message = message + "-" * 20 + "Options" + "-" * 20 + "\n"
        arg_groups = self.parser._action_groups
        for group in arg_groups:
            _args = group._group_actions
            if len(_args) != 0:
                message = message + "-" * 5 + group.title + "-" * 5 + "\n"
                for arg in _args:
                    arg_name = arg.dest
                    arg_value = vars(self.args)[arg_name]
                    default_value = self.parser.get_default(arg_name)
                    remark = ""
                    if arg_value != default_value:
                        remark = '\t(default: {})\t'.format(default_value)
                    message = message + "{}: {}{}\n".format(arg_name, arg_value, remark)
        message = message + "-" * 20 + "End" + "-" * 20
        print(self.parser.description)
        print(message)

    def _set_gpu(self):
        gpu_ids = self.args.gpu_ids.split(',')
        gpu_ids = [ int(gpu_id) for gpu_id in gpu_ids]
        return gpu_ids

