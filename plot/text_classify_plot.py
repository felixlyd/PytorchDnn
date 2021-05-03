import json
import torch

from plot.base_plot import BasePlot


class TextClassifierPlot(BasePlot):
    def __init__(self, opt):
        super(TextClassifierPlot, self).__init__(opt)

    def write_information(self):
        if self.plot:
            information = json.dumps(vars(self.opt))
            self.writer.add_text("My-PyDNN/TextClassifierParams", information)