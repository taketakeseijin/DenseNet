
import torch.nn as nn

import parts


def get_model(config):
    growth_rate = config["k"]
    depth = config["depth"]
    n_class = config["class"]
    use_bottleneck = config["BC"]
    if use_bottleneck:
        reduction_rate = 0.5
    else:
        reduction_rate = 1

    model = DenseNet(growth_rate, depth, reduction_rate,
                     n_class, use_bottleneck)
    return model


class DenseNet(nn.Module):
    def __init__(self, growth_rate, depth, reduction_rate, n_class, use_bottleneck=True):
        super().__init__()
        self.k = growth_rate
        self.depth = depth
        self.theta = reduction_rate
        self.n_class = n_class
        self.use_bottleneck = use_bottleneck

        # 4 means 1 conv and 3 trans layer
        dense_layer_depth = (self.depth - 4) // 3
        if self.use_bottleneck:
            dense_layer_depth = dense_layer_depth // 2

        nChannels = 2 * self.k
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=nChannels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.dense_transition1, out_channels = parts.make_dense_transition(
            in_channels=nChannels,
            growth_rate=self.k,
            reduction_rate=self.theta,
            depth=dense_layer_depth,
            use_bottleneck=self.use_bottleneck,
            bias=False
        )
        nChannels = out_channels

        self.dense_transition2, out_channels = parts.make_dense_transition(
            in_channels=nChannels,
            growth_rate=self.k,
            reduction_rate=self.theta,
            depth=dense_layer_depth,
            use_bottleneck=self.use_bottleneck,
            bias=False
        )
        nChannels = out_channels

        self.dense3, out_channels = parts.make_dense(
            in_channels=nChannels,
            growth_rate=self.k,
            depth=dense_layer_depth,
            use_bottleneck=self.use_bottleneck,
            bias=False
        )
        nChannels = out_channels

        self.pool = nn.Sequential(
            nn.BatchNorm2d(nChannels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(
            in_features=nChannels,
            out_features=self.n_class
        )

    def forward(self, input_x):
        # input_x shape should [batch, 3, 32, 32]
        out = self.conv(input_x)
        out = self.dense_transition1(out)
        out = self.dense_transition2(out)
        out = self.dense3(out)
        out = self.pool(out).squeeze()
        out = self.fc(out)
        return out
