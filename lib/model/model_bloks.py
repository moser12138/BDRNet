import torch
from torch import nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

# 基础残差块，通道数不定，size不变
class Res_basic(nn.Module):
    def __init__(self, in_channels, out_channels, no_relu=False):
        super(Res_basic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels, momentum=bn_mom)
        self.no_relu = no_relu

        # 调整输入张量的维度，以便能够与残差块的输出相加
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                BatchNorm2d(out_channels, momentum=bn_mom)
            )
            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                BatchNorm2d(out_channels, momentum=bn_mom)
            )
        else:
            self.shortcut = nn.Identity()
            self.shortcut2 = nn.Identity()

    def forward(self, x, res=None):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)

        if res is None:
            res = x
        else:
            res = self.shortcut2(res) + x

        x = self.relu(res)
        res = x
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity

        if self.no_relu:
            return x, res
        else:
            return self.relu(x), res

# 瓶颈残差块,输出通道数为out_channels的两倍
class Res_bottle(nn.Module):
    def __init__(self, in_channels, mid_channels, outchannels, stride=1, no_relu=True):
        super(Res_bottle, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(mid_channels, momentum=bn_mom)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(mid_channels, momentum=bn_mom)
        self.conv3 = nn.Conv2d(mid_channels, outchannels, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(outchannels, momentum=bn_mom)
        self.residual = nn.Sequential(
                nn.Conv2d(in_channels, outchannels, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(outchannels, momentum=bn_mom)
            )
        if in_channels != mid_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
                BatchNorm2d(mid_channels, momentum=bn_mom)
            )
        else:
            self.shortcut = nn.Identity()
        # if stride != 1:
        #     self.downsample = nn.Sequential(
        #         nn.Conv2d(in_channels, outchannels, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(outchannels, momentum=bn_mom),
        #     )
        # else:
        #     self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res):
        residual = self.residual(x)
        res = self.shortcut(res)

        x = self.conv1(x)
        x = self.bn1(x)
        x += res
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += residual
        x = self.relu(x)

        return x

# 残差下采样，通道数不定，size减半
class Res_downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_downsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels, momentum=bn_mom)

        # 调整输入张量的维度，以便能够与残差块的输出相加

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom)
        )

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(in_channels, momentum=bn_mom)
        )
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom)
        )

    def forward(self, x, res=None):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)

        if res is not None:
            pool = self.pool(res)
            res = self.res_conv1(res)
            res = torch.cat([pool, res], dim=1)
            res = self.res_conv2(res)
            x = res + x

        x = self.relu(x)
        res = x
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)

        return x, res

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out

class DRM(nn.Module):
    def __init__(self, channels_x, channels_y):
        super(DRM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.ChangeChannel = nn.Sequential(
            nn.Conv2d(in_channels=channels_y, out_channels=channels_x, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_x)
        )
        self.MainBranch_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels_x, out_channels=channels_x, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_x)
        )
        self.DetailBranch_1 = nn.Sequential(
            nn.Conv2d(channels_x, channels_x, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_x)
        )

    def forward(self, x, y):
        input_size = x.size()
        x_1 = self.MainBranch_1(x)

        y = self.ChangeChannel(y)
        y_1 = self.DetailBranch_1(y)
        y_1 = F.interpolate(y_1, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)

        sig_map = torch.sigmoid(torch.sum(x_1 * y_1, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x = (1 - sig_map) * x + sig_map * y
        x = self.relu(x)
        return x

class ConCact(nn.Module):
    def __init__(self, inchannels, outchannels, group):
        super(ConCact, self).__init__()
        self.Detail_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1, groups=group, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.Detail_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=2, padding=1, bias=False),  # 下采样
            nn.BatchNorm2d(outchannels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)  # 下采样1/32
        )
        self.Main_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannels),
        )
        self.Main_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1, groups=group, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=8)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up3 = nn.Upsample(scale_factor=4)

        self.last = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),  # not shown in paper
        )

    def forward(self, x_d, x_m):  #1/8,1/64
        xd_1 = self.Detail_1(x_d)
        xd_2 = self.Detail_2(x_d)

        xm_1 = self.Main_1(x_m)
        xm_2 = self.Main_2(x_m)
        xm_2 = self.up2(xm_2)

        xm_1 = self.up1(xm_1)
        xd = xd_1 * torch.sigmoid(xm_1)

        xm = xd_2 * torch.sigmoid(xm_2)
        xm = self.up3(xm)

        out = self.last(xd + xm)
        return out


if __name__ == '__main__':
    # img1 = torch.randn(2, 16, 1024, 2048).cuda()
    imgspp = torch.randn(2, 512, 16, 32).cuda()

    # net1 = Res_basic(16, 16).cuda()
    # net2 = Res_basic(16, 32).cuda()

    print(outspp.size())