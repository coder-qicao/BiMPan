import torch
import torch.nn as nn


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        # if downsample:
            # self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        # if self.downsample:
        #     out_down = self.downsample(out)
        #     return out_down, out
        else:
            return out


class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, channels, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main_fft = nn.Sequential(
            nn.Conv2d(2*channels, 2*channels, kernel_size=3, stride=1, padding=1, groups=2*channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*channels, 2*channels, kernel_size=3, stride=1, padding=1, groups=2*channels),)
        self.merging = UNetConvBlock(in_size=2*channels, out_size=channels, downsample=False, relu_slope=0.2, use_csff=False, use_HIN=True)
        self.norm = norm
    def forward(self, x):  # 1, 32, 64, 64
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)  # 1, 32, 64, 63
        y_imag = y.imag  # 实部 1, 32, 64, 63
        y_real = y.real  # 虚部 1, 32, 64, 63
        y_f = torch.cat([y_real, y_imag], dim=dim) # 1, 64, 64, 63
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        out = self.merging(torch.cat([x, y], dim=1))
        return out

class ResBlock_do_fft_bench2(nn.Module):
    def __init__(self, channels, norm='backward'):
        super(ResBlock_do_fft_bench2, self).__init__()

        self.fft_conv1 = nn.Sequential(
            nn.Conv2d(2*channels, 2*channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, stride=1),
        )
        self.merging = UNetConvBlock(in_size=2*channels, out_size=channels, downsample=False, relu_slope=0.2, use_csff=False, use_HIN=True)
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.fft_conv1(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        out = self.merging(torch.cat([x, y], dim=1))
        return out

class FGMM(nn.Module):
    def __init__(self, dim):
        super(FGMM, self).__init__()
        self.fourier_spatial = ResBlock_do_fft_bench(dim)
        self.fourier_spectral = ResBlock_do_fft_bench2(dim)

    def forward(self, x):
        x = self.fourier_spatial(x)
        out = self.fourier_spectral(x)
        return out