from architectures.u_net.baseclass.UNet import UNet
from torch.nn import Conv2d

class Deterministic(UNet):
    def __init__(self, nfilters, kernel_size):
        super().__init__(nfilters, kernel_size, Conv2d)
