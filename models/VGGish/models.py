import sys

import numpy as np
import torch
import config
sys.path.insert(0, str(config.LIBS_DIR.joinpath('resnet1d')))
from resnet1d import ResNet1D

#Wavelet stuff
from scipy import signal

class ResNet1DModel(torch.nn.Module):
        def __init__(self, out_channels=2):
            super(ResNet1DModel, self).__init__()
            self.resnet = ResNet1D(in_channels=1,
                                    base_filters=64,
                                    kernel_size=16,
                                    stride=2,
                                    groups=32,
                                    n_block=36,
                                    n_classes=out_channels,
                                    downsample_gap=6,
                                    increasefilter_gap=12,
                                    use_bn=True,
                                    use_do=True,
                                    verbose=False).cuda()
            
        def forward(self, x):
            x = torch.mean(x, dim=1, keepdim=True)
            x = self.resnet.forward(x)
            return x  

def get_resnet1d_model(out_channels=2):
    resnet_model = ResNet1DModel(out_channels=out_channels)
    return resnet_model