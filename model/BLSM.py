import torch
import torch.nn as nn
from torch.nn import functional as F
from model.models_others import SoftAttn


class AConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=False):
        super(AConv, self).__init__()
        self.in_features = in_planes
        self.out_features = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        # self.ch_att = ChannelAttention(kernel_size ** 2)  # change

        # Generating local adaptive weights
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_size ** 2, kernel_size, stride, padding),
            SoftAttn(kernel_size ** 2)
        )
        conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, x):
        (b, n, H, W) = x.shape
        o_f = self.out_features
        k = self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        att1 = self.attention1(x)  # b,k*k,n_H,n_W
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        att1 = att1.permute([0, 2, 3, 1])  # b,n_H,n_W,k*k
        att1 = att1.unsqueeze(3).repeat([1, 1, 1, n, 1])  # b,n_H,n_W,n,k*k
        att1 = att1.view(b, n_H, n_W, n * k * k)  # b,n_H,n_W,n*k*k

        att2 = att1  # *att2 #b,n_H,n_W,n*k*k
        att2 = att2.view(b, n_H * n_W, n * k * k)  # b,n_H*n_W,n*k*k
        att2 = att2.permute([0, 2, 1])  # b,n*k*k,n_H*n_W

        kx = F.unfold(x, kernel_size=k, stride=self.stride, padding=self.padding)  # b,n*k*k,n_H*n_W
        atx = att2 * kx  # b,n*k*k,n_H*n_W

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k
        atx = atx.view(1, b * n_H * n_W, n * k * k)  # 1,b*n_H*n_W,n*k*k

        w = self.weight.view(o_f, n * k * k)  # m,n*k*k
        w = w.permute([1, 0])  # n*k*k,m
        y = torch.matmul(atx, w)  # 1,b*n_H*n_W,m
        y = y.view(b, n_H * n_W, o_f)  # b,n_H*n_W,m

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W
        return y


class SDAConv(nn.Module):
    def __init__(self, out_channels):
        super(SDAConv, self).__init__()
        self.conv1 = AConv(out_channels, out_channels, 3, 1, 1, use_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = AConv(out_channels, out_channels, 3, 1, 1, use_bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        x = x + res
        return x


class SDA_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SDA_Block, self).__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ADT1 = SDAConv(out_channels)
        self.ADT2 = SDAConv(out_channels)
        self.ADT3 = SDAConv(out_channels)

    def forward(self, p, s):
        f_p = self.conv_p(p)
        f_s = self.conv_s(s)
        f_in = self.conv(torch.cat([f_p, f_s], dim=1))
        out1 = self.ADT1(f_in)
        out2 = self.ADT2(out1)
        out = self.ADT3(out2)
        return out