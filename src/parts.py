
from collections import OrderedDict
import math
import torch
import torch.nn as nn

class BottleneckSlice(nn.Module):
    def __init__(self,in_channels,growth_rate,bias):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.inner_channels = 4 * self.growth_rate

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inner_channels,
                kernel_size=1,
                bias=bias
                ),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.inner_channels,
                out_channels=self.growth_rate,
                kernel_size=3,
                padding=1,
                bias=bias
                ),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat([x,out],1)

class NormalSlice(nn.Module):
    def __init__(self,in_channels,growth_rate,bias):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.growth_rate,
                kernel_size=3,
                padding=1,
                bias=bias,
                ),
        )

    def forward(self, x):
        out = self.conv1(x)
        return torch.cat([x,out],1)


def make_dense(in_channels,growth_rate,depth,use_bottleneck,bias):
    dense_ODict = OrderedDict()
    if use_bottleneck:
        dense_slice_class = BottleneckSlice
    else:
        dense_slice_class = NormalSlice
    for d in range(depth):
        dense_ODict[f"slice:{d}"] = dense_slice_class(in_channels,growth_rate,bias)
        in_channels += growth_rate
    dense_layer = nn.Sequential(dense_ODict)
    return dense_layer, in_channels

def make_transition(in_channels,reduction_rate,bias):
    out_channels = int(math.floor(in_channels*reduction_rate))
    conv_kernel = 1
    pool_kernel = 2
    pool_stride = 2
    transition = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel,
            bias=bias
            ),
        nn.AvgPool2d(pool_kernel,pool_stride)
        )
    return transition, out_channels

def make_dense_transition(in_channels,growth_rate,reduction_rate,depth,use_bottleneck,bias):
    dense_layer,in_channels = make_dense(in_channels,growth_rate,depth,use_bottleneck,bias)
    transition,out_channels = make_transition(in_channels,reduction_rate,bias)
    return nn.Sequential(dense_layer,transition),out_channels