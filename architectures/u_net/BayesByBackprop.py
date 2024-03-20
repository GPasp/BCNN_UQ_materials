from enum import Enum

import torch.nn.functional as F
from torch.nn import ModuleList
import numpy as np
from .baseclass import UNet
from .layers.BBBConv import BBBConv2d as BBBConv2d
from .layers.BBBConv_LRT import BBBConv2d as BBBConv_LRT


class BayesLayerType(Enum):
    BBB = 1
    BBB_LRT = 2

class BayesByBackprop(UNet):
    def __init__(self, nfilters, kernel_size, layer_type):
        if layer_type == 'BBB':
            layer_type = BBBConv2d
        elif layer_type == 'BBB_LRT':
            layer_type = BBBConv_LRT
        else:
            raise ValueError(f"Layer type {layer_type} not recognized")
        super().__init__(nfilters, kernel_size, layer_type)

    def get_kl_loss_layers(self):
        model = self
        #  the KL loss
        kl_loss_total = 0.0
        # Recursively calculate the KL loss for all submodules that have the 'kl_loss' attribute
        for module in model.modules():
            if hasattr(module, 'kl_loss'):
                kl_loss_total += module.kl_loss()
        return kl_loss_total
