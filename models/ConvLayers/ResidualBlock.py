import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()

        self.first_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        for _ in range(num_layers):
            self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels

        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.first_conv(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)

        x = self.layers[-1](x)
        x = x + x0
        x = self.relu(x)

        return x
