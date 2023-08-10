from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=7):
        super(Eca_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # change
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.conv1 = nn.Conv2d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 保持卷积前后H、W不变 change
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        # y1 = self.avg_pool(x)
        ########## change
        y2 = self.max_pool(x)  # change
        y =  y2  # change

        # x = torch.cat([y1, y2], dim=1)
        # print(x.shape)
        # y = self.conv1(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y

        # return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ration=16):
        super(ChannelAttention, self).__init__()

        '''
        AdaptiveAvgPool2d():自适应平均池化
                            不需要自己设置kernelsize stride等
                            只需给出输出尺寸即可
        '''
        k_size = 7
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 通道数不变，H*W变为1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # change
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print(avg_out.shape)
        # 两层神经网络共享
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # print(avg_out.shape)
        # print(max_out.shape)
        out = avg_out + max_out

        # out = self.conv(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # change
        # print(out.shape)
        return self.sigmoid(out)


''''
空间注意力模块
        先分别进行一个通道维度的最大池化和平均池化得到两个H x W x 1，
        然后两个描述拼接在一起，然后经过一个7*7的卷积层，激活函数为sigmoid，得到权重Ms

'''


class SpatialAttention(nn.Module):
    def __init__(self, in_planes, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), " kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # avg 和 max 两个描述，叠加 共两个通道。
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 保持卷积前后H、W不变
        self.sigmoid = nn.Sigmoid()

        # self.avgpool = nn.AvgPool2d(2, stride=2)  #### change ##############
        # self.conv2 = nn.Conv2d(in_planes, 1, kernel_size, padding=padding, bias=False)  #### change ##############

    def forward(self, x):
        # (b, n, H, W) = x.shape
        # y1 = self.avgpool(x)  #### change ##############
        # y2 = self.conv2(y1)  # change  1 channel
        # t = nn.Upsample(scale_factor=2)  #### change ##############
        # y2 = t(y2)  #### change ##############

        # egg：input: 1 , 3 * 2 * 2  avg_out :
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道维度的平均池化
        # 注意 torch.max(x ,dim = 1) 返回最大值和所在索引，是两个值  keepdim = True 保持维度不变（求max的这个维度变为1），不然这个维度没有了
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度的最大池化
        # print(avg_out.shape)
        # print(max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        # print(x.shape)
        x = self.conv1(x)
        # x = x + y2  # change
        return self.sigmoid(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=5, last=nn.ReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        elif kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            last()
        )

    def forward(self, x):
        out = self.main(x)
        return out


class SoftAttn(nn.Module):  # 软注意力（32，16，256，128）=空间注意力的输出（32，1，256，128）乘上通道注意力（32,16,1,1）

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttention(in_channels)
        # self.channel_attn = ChannelAttention(in_channels)
        self.channel_attn = Eca_layer(in_channels)
        self.conv = ConvLayer(in_channels, in_channels, 3)

    def forward(self, x):  # x.shape(32,16,256,128)
        y_spatial = self.spatial_attn(x)  # 32,1,256,128
        y_channel = self.channel_attn(x)  # 32,16,1,1
        y = y_spatial * y_channel  # 32,16,256,128
        y = F.sigmoid(self.conv(y))
        return y  # torch.Size([32, 16, 256, 128])


class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, batch_size, num_layers):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size  # 这个为词向量的维数
        self.hidden_size = hidden_size  # 隐藏单元数
        self.output_size = output_size  # 最后要输出的
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = nn.RNN(self.vocab_size, self.hidden_size,
                          batch_first=True, num_layers=self.num_layers, dropout=0.5)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个,
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, x, x_lens):
        # 每一个batch都要重新初始化hidden、cell，不然模型会将上一个batch的hidden、cell作为初始值
        self.hidden = self.init_hidden()
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(x_packed, self.hidden)
        output, lens = pad_packed_sequence(output, batch_first=True)
        output = torch.Tensor(
            [output[i, (lens[i] - 1).item(), :].tolist() for i in range(self.batch_size)])
        output = self.linear(output)
        return F.log_softmax(output, dim=1)


class LAConv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(LAConv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        self.ch_att = ChannelAttention(kernel_size ** 2)  # change

        # Generating local adaptive weights
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_size ** 2, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),  # changed
            nn.Sigmoid()  # changed
        )  # b,9,H,W È«Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W µ¥Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        self.spatt = SpatialAttention(3)
        if use_bias == True:  # Global local adaptive weights
            self.attention3 = nn.Sequential(  # like channel attention
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes, out_planes, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, 1)  # change
            )  # b,m,1,1 Í¨µÀÆ«ÖÃ×¢ÒâÁ¦

        conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, x):
        (b, n, H, W) = x.shape
        m = self.out_planes
        k = self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1 = self.attention1(x)  # b,k*k,n_H,n_W
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1 = atw1.permute([0, 2, 3, 1])  # b,n_H,n_W,k*k
        atw1 = atw1.unsqueeze(3).repeat([1, 1, 1, n, 1])  # b,n_H,n_W,n,k*k
        atw1 = atw1.view(b, n_H, n_W, n * k * k)  # b,n_H,n_W,n*k*k

        # atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw = atw1  # *atw2 #b,n_H,n_W,n*k*k
        atw = atw.view(b, n_H * n_W, n * k * k)  # b,n_H*n_W,n*k*k
        atw = atw.permute([0, 2, 1])  # b,n*k*k,n_H*n_W

        kx = F.unfold(x, kernel_size=k, stride=self.stride, padding=self.padding)  # b,n*k*k,n_H*n_W
        atx = atw * kx  # b,n*k*k,n_H*n_W

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k
        atx = atx.view(1, b * n_H * n_W, n * k * k)  # 1,b*n_H*n_W,n*k*k

        w = self.weight.view(m, n * k * k)  # m,n*k*k
        w = w.permute([1, 0])  # n*k*k,m
        y = torch.matmul(atx, w)  # 1,b*n_H*n_W,m
        y = y.view(b, n_H * n_W, m)  # b,n_H*n_W,m
        if self.bias == True:
            bias = self.attention3(x)  # b,m,1,1
            bias = bias.view(b, m).unsqueeze(1)  # b,1,m
            bias = bias.repeat([1, n_H * n_W, 1])  # b,n_H*n_W,m
            y = y + bias  # b,n_H*n_W,m

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.net(x).view(batch_size, 1)
        x2 = torch.sigmoid(x1)
        return torch.sigmoid(x1)


# LAC_ResBlocks
class LACRB(nn.Module):
    def __init__(self, in_planes):
        super(LACRB, self).__init__()
        self.conv1 = LAConv2D(in_planes, in_planes, 3, 1, 1, use_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = LAConv2D(in_planes, in_planes, 3, 1, 1, use_bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        x = x + res
        return x
