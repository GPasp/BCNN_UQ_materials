from torch.nn import Conv2d, MaxPool2d, Module, BatchNorm2d, Upsample
from torch.nn import Dropout2d
from .baseclass import UNet

class MonteCarloDropout(UNet):
    def __init__(self, nfilters, kernel_size, drop_rate, layer_drop_idx_en, layer_drop_idx_dec):
        super().__init__(nfilters, kernel_size, Conv2d)
        # self.drop_layer_index = drop_layer_index
        self.drop_rate = drop_rate
        self.dropout = Dropout2d(self.drop_rate)
        self.layer_drop_idx_en = layer_drop_idx_en
        self.layer_drop_idx_dec = layer_drop_idx_dec

    def optional_step_en(self, x, i):
        if self.layer_drop_idx_en[i] == 1:
            x = self.dropout(x)
        return x

    def optional_step_dec(self, x, i):
        if self.layer_drop_idx_dec[i] == 1:
            x = self.dropout(x)
        return x
