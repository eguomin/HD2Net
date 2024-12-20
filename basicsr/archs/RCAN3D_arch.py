import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', activation=None):
        super(ConvBlock3D, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x

class GlobalAveragePooling3D(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)

class ChannelAttentionBlock3D(nn.Module):
    def __init__(self, num_channels, reduction):
        super(ChannelAttentionBlock3D, self).__init__()
        self.global_avg_pool = GlobalAveragePooling3D()
        self.fc1 = ConvBlock3D(num_channels, num_channels // reduction, 1, activation='relu')
        self.fc2 = ConvBlock3D(num_channels // reduction, num_channels, 1, activation='sigmoid')

    def forward(self, x):
        y = self.global_avg_pool(x)
        y = y.view(y.size(0), y.size(1), 1, 1, 1)
        y = self.fc1(y)
        y = self.fc2(y)
        return x * y

class ResidualChannelAttentionBlock3D(nn.Module):
    def __init__(self, num_channels, reduction, residual_scaling):
        super(ResidualChannelAttentionBlock3D, self).__init__()
        self.conv1 = ConvBlock3D(num_channels, num_channels, 3, activation='relu')
        self.conv2 = ConvBlock3D(num_channels, num_channels, 3)
        self.channel_attention = ChannelAttentionBlock3D(num_channels, reduction)
        self.residual_scaling = residual_scaling

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.channel_attention(x)
        x = x * self.residual_scaling
        x += residual
        return x

class ResidualGroup3D(nn.Module):
    def __init__(self, num_channels, num_blocks, reduction, residual_scaling):
        super(ResidualGroup3D, self).__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualChannelAttentionBlock3D(num_channels, reduction, residual_scaling))
        self.blocks = nn.Sequential(*blocks)
        self.conv = ConvBlock3D(num_channels, num_channels, 3)

    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x = self.conv(x)
        x += residual
        return x

@ARCH_REGISTRY.register()
class RCAN3D(nn.Module):
    def __init__(self, input_channel, num_channels, num_blocks, num_groups, reduction, residual_scaling, num_output_channels):
        super(RCAN3D, self).__init__()
        self.conv1 = ConvBlock3D(input_channel, num_channels, 3)
        self.residual_groups = nn.Sequential(
            *[ResidualGroup3D(num_channels, num_blocks, reduction, residual_scaling) for _ in range(num_groups)]
        )
        self.conv2 = ConvBlock3D(num_channels, num_channels, 3)
        self.conv3 = ConvBlock3D(num_channels, num_output_channels, 3)

    def forward(self, x):
        x = x * 2 - 1  # Standardize
        long_skip = self.conv1(x)
        x = self.residual_groups(long_skip)
        x = self.conv2(x)
        x += long_skip
        x = self.conv3(x)
        x = x * 0.5 + 0.5  # Destandardize
        return x



# if __name__ == '__main__':
#
#     model = RCAN3D(1, num_channels=32, num_blocks=3, num_groups=5, reduction=8,
#                    residual_scaling=1.0, num_output_channels=1).cuda()
#     input = torch.rand(1, 1, 64, 64, 64).cuda()
#     output = model(input)
#     print('===================')
#     print(output.shape)

