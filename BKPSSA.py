import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class ConvAvgPool(nn.Module):
    def __init__(self, in_channels, out_channels, is_bn=False, is_UpSample=False):
        super(ConvAvgPool, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
        self.is_bn = is_bn
        if self.is_bn:
            self.bn = nn.BatchNorm2d(out_channels)
            self.bn.weight.data.fill_(1)
        if is_UpSample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        if self.is_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x, self.conv.weight


class ConvResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_bn=False):
        super(ConvResnetBlock, self).__init__()
        self.is_bn = is_bn
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.is_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class Unsqueeze(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unsqueeze, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x, self.conv.weight


class BKPUnet(nn.Module):
    def __init__(self, in_channels, out_channels, Sample_number=3, Middle_number=4, K=5):
        super(BKPUnet, self).__init__()
        self.Sample_number = Sample_number
        self.Middle_number = Middle_number
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DownSample = nn.ModuleList([])
        self.UpSample = nn.ModuleList([])
        self.Middle = nn.ModuleList([])
        for i in range(self.Sample_number):
            in_c = self.in_channels * (2 ** i)
            out_c = self.out_channels * (2 ** (i + 1))
            self.DownSample.append(ConvAvgPool(in_c, out_c))
            self.UpSample.append(ConvAvgPool(out_c, in_c))

        c_m = self.out_channels * (2 ** (self.Sample_number - 1))
        for i in range(self.Middle_number):
            self.Middle.append(ConvResnetBlock(c_m, c_m))

        self.Unsqueeze = Unsqueeze(c_m, c_m)

    def forward(self, x_in):
        x_list = []
        weight_list = []
        x = x_in
        weight = Parameter(torch.ones([1, 1, self.K, self.K]))
        for i in range(self.Sample_number):
            x, weight = self.DownSample[i](x)
            x_list.append(x)
            weight = weight * weight_list[i][1, 1, self.K, self.K]
            weight_list.append(weight)

        for i in range(self.Middle_number):
            x = self.Middle[i](x)
            if i == self.Middle_number - 1:
                x, weight = self.Unsqueeze(x)
                weight_list.append(weight)

        for i in range(self.Sample_number):
            x = self.UpSample[i](x)
            x = x + x_list[self.Sample_number - i - 1]
            weight = weight * weight_list[self.Sample_number - i]

        return x + x_in, weight


## Shallow Feature Extract
class hybrid_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(hybrid_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.relu4 = nn.LeakyReLU(inplace=True)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.relu1(x)
        x1 = x.unsqueeze(0)
        x2 = x1.permute(0, 2, 1, 3, 4)
        x1 = self.relu2(self.conv3(x1))
        x2 = self.relu3(self.conv3(x2))
        x = x1.squeeze(1)+x2.squeeze(2)
        x = self.relu4(self.conv4(x))
        return x+x_in


class SFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SFE, self).__init__()
        self.ln = nn.LayerNorm([in_channels, 1, 1])
        self.hybrid = hybrid_conv(in_channels, out_channels)
        self.resnet = ConvResnetBlock(in_channels, out_channels)
        self.add_norm = nn.LayerNorm([out_channels, 1, 1])

    def forward(self, x_in):
        x = self.ln(x_in)
        x = self.hybrid(x)
        x = self.resnet(x)
        x = self.add_norm(x + x_in)
        return x


## Deep Feature Extract
class SpectralAttention(nn.Module):
    """光谱注意力模块，关注光谱维度上的特征"""

    def __init__(self, channels, reduction=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块，关注空间维度上的特征"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)

        return x * self.sigmoid(y)


class SelfAttention(nn.Module):
    """自注意力模块，捕捉空间位置之间的长距离依赖关系"""

    def __init__(self, in_channels, key_channels=None, value_channels=None, out_channels=None, scale=1):
        super(SelfAttention, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels or in_channels // 8
        self.value_channels = value_channels or in_channels
        self.out_channels = out_channels or in_channels

        # 查询、键、值投影
        self.conv_query = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, self.value_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(self.value_channels, self.out_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # 为了降低计算复杂度，可以在空间维度上进行下采样
        if self.scale > 1:
            x = F.avg_pool2d(x, kernel_size=self.scale, stride=self.scale)

        # 生成查询、键、值
        query = self.conv_query(x).view(batch_size, self.key_channels, -1)
        key = self.conv_key(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        value = self.conv_value(x).view(batch_size, self.value_channels, -1)

        # 计算注意力分数
        sim_map = torch.bmm(key, query)  # [B, HW, HW]
        sim_map = self.softmax(sim_map)

        # 应用注意力
        context = torch.bmm(value, sim_map)  # [B, C, HW]
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.conv_out(context)

        # 如果之前做了下采样，需要上采样回原始大小
        if self.scale > 1:
            context = F.interpolate(context, size=(height, width), mode='bilinear', align_corners=True)

        # 残差连接
        out = self.gamma * context + x

        return out


class SpectralSpatialSelfAttention(nn.Module):
    """结合光谱、空间和自注意力的模块"""

    def __init__(self, channels=128, reduction=16, spatial_kernel_size=7, self_attention_scale=2):
        super(SpectralSpatialSelfAttention, self).__init__()

        # 光谱特征提取
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # 注意力模块
        self.spectral_attention = SpectralAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        self.self_attention = SelfAttention(channels, scale=self_attention_scale)

    def forward(self, x):
        # 输入 x 的形状为 (1, 128, 128, 128)

        # 初始特征提取
        out = self.conv_3x3(x)
        out = self.bn(out)
        out = self.relu(out)

        # 应用光谱注意力
        out = self.spectral_attention(out)

        # 应用空间注意力
        out = self.spatial_attention(out)

        # 应用自注意力
        out = self.self_attention(out)

        # 残差连接
        out = out + x

        return out


class SSAB(nn.Module):
    """带自注意力机制的高光谱图像特征提取网络"""

    def __init__(self, input_channels=128, num_blocks=3):
        super(SSAB, self).__init__()

        self.input_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)

        # 创建多个空间-光谱-自注意力模块
        self.attention_blocks = nn.ModuleList([
            SpectralSpatialSelfAttention(channels=input_channels)
            for _ in range(num_blocks)
        ])

        self.output_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)

    def forward(self, x):
        """
        输入: (batch_size, channels, height, width) = (1, 128, 128, 128)
        输出: (batch_size, channels, height, width) = (1, 128, 128, 128)
        """
        identity = x

        x = self.input_conv(x)

        # 依次通过多个注意力模块
        for block in self.attention_blocks:
            x = block(x)

        x = self.output_conv(x)

        # 最终残差连接
        out = x + identity

        return out


class SSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SSAB = SSAB(self.in_channels, self.out_channels)

    def forward(self, x_in):
        x = self.SSAB(x_in)
        return x + x_in
