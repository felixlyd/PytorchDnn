import json
import torch

from plot.base_plot import BasePlot


class TextClassifierPlot(BasePlot):
    def __init__(self, opt):
        super(TextClassifierPlot, self).__init__(opt)

    def write_model(self, model_, device, vocab_size, seq_len):
        input_tensor = torch.randint(0, vocab_size, [100, seq_len]).to(device)
        if self.plot:
            self.writer.add_graph(model_, input_tensor)

    def write_information(self):
        if self.plot:
            information = json.dumps(vars(self.opt))
            self.writer.add_text("My-PyDNN/TextClassifierParams", information)
