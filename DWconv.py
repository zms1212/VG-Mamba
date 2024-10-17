import torch
import torch.nn as nn

# 定义深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度可分离卷积包括深度卷积和逐点卷积两个步骤

        if in_channels == 3:
            hidden_features = 8
        else:
            hidden_features = out_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        # self.proj = Partial_conv3(hidden_features, hidden_features, 'split_cat')
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_channels, eps=1e-5),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.proj(x)
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        return x




