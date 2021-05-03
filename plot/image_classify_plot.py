import json
import torch

from plot.base_plot import BasePlot


class ImageClassifierPlot(BasePlot):
    def __init__(self, opt):
        super(ImageClassifierPlot, self).__init__(opt)

    def write_model(self, model_, device, input_size):
        input_tensor = torch.rand([1, 3, input_size, input_size]).to(device)
        if self.plot:
            self.writer.add_graph(model_, input_tensor)

    def write_information(self):
        if self.plot:
            information = json.dumps(vars(self.opt))
            self.writer.add_text("My-PyDNN/ImageClassifierParams", information)
