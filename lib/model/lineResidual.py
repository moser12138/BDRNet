import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

from torch.nn import ReLU as relu

from .model_bloks import bn_mom
from .model_bloks import BatchNorm2d
from .model_bloks import Res_basic
from .model_bloks import Res_bottle
from .model_bloks import Res_downsample
from .model_bloks import DAPPM

# inter_mode = 'neraest'
inter_mode = 'bilinear'
backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'

class ConvBnRelu(nn.Module):
    # 卷积，批标准化，激活三合一模块
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,padding=padding, dilation=dilation,groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class segmenthead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(segmenthead, self).__init__()
        self.conv = ConvBnRelu(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes

        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBnRelu(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat
class RDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RDown, self).__init__()
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

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        res = x
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)

        return x, res

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
# class segmenthead(nn.Module):
#
#     def __init__(self, inplanes, interplanes, outplanes, height, weight):
#         super(segmenthead, self).__init__()
#         self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
#         self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
#         self.height = height
#         self.weight = weight
#
#     def forward(self, x):
#         x = self.conv1(self.relu(self.bn1(x)))
#         out = self.conv2(self.relu(self.bn2(x)))
#         out = F.interpolate(out, size=[self.height, self.weight], mode='bilinear')
#
#         return out
class Liner_Risidual(nn.Module):
    def __init__(self, n_classes, aux_mode='train'):
        super(Liner_Risidual, self).__init__()
        self.aux_mode = aux_mode #模型类型
        self.relu = relu(inplace=True)
        self.bn = BatchNorm2d(64, momentum=bn_mom)

        # 1/4
        self.s11 = RDown(3, 32)
        # self.s11 = Res_downsample(3, 32)
        self.s12 = Res_downsample(32, 32)
        self.s13 = Res_basic(32, 64)

        # 1/8
        self.s31 = Res_downsample(64, 64)
        self.s32 = Res_basic(64, 64)

        # 1/16 1/8
        self.s41 = Res_downsample(64, 128)
        self.s42 = Res_basic(128, 128)
        self.s41_ = Res_basic(64, 64)
        self.compression4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            BatchNorm2d(64, momentum=bn_mom),
        )
        self.compression4_res = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            BatchNorm2d(64, momentum=bn_mom),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(128, momentum=bn_mom),
        )
        self.down4_res = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(128, momentum=bn_mom),
        )

        # 1/32 1/8
        self.s51 = Res_downsample(128, 256)
        self.s52 = Res_basic(256, 256)
        self.s51_ = Res_basic(64, 64)
        self.compression5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            BatchNorm2d(64, momentum=bn_mom),
        )
        self.compression5_res = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            BatchNorm2d(64, momentum=bn_mom),
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(128, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(256, momentum=bn_mom),
        )
        self.down5_res = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(128, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(256, momentum=bn_mom),
        )

        # 1/32 1/8
        self.s61 = Res_bottle(256, 256, 512, stride=2)
        self.s61_ = Res_bottle(64, 64, 64)
        self.spp = DAPPM(512, 128, 64)

        # self.concact = ConCact(64, 64, 32)

        self.seghead = segmenthead(64, 256, n_classes, up_factor=8, aux=False)
        # self.seghead = segmenthead(64, 128, n_classes, 1024, 1024)
        if self.aux_mode == 'train':
            # self.aux_loss = segmenthead(64, 128, n_classes, 1024, 1024)
            self.aux_loss = segmenthead(64, 128, n_classes, 8)

    def forward(self, x):
        global aux
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        # 1/2
        x, res = self.s11(x)
        x, res = self.s12(x, res)
        x, res = self.s13(x, res)
        # 1/8
        x, res = self.s31(x, res)
        x, res = self.s32(x, res)
        # 1/16 1/8 -> x x_
        x_, res_ = self.s41_(x, res)
        x, res = self.s41(x, res)
        x, res = self.s42(x, res)

        middle = F.interpolate(self.compression4(x), size=[height_output, width_output], mode=inter_mode)
        x = self.relu(x + self.down4(x_))
        x_ = self.relu(x_ + middle)
        if self.aux_mode == 'train':
            aux = x_

        middle = F.interpolate(self.compression4_res(res), size=[height_output, width_output], mode=inter_mode)
        res = self.relu(res + self.down4_res(res_))
        res_ = self.relu(res_ + middle)

        # 1/32 1/8 -> x x_
        x, res = self.s51(x, res)
        x, res = self.s52(x, res)
        x_, res_ = self.s51_(x_, res_)

        middle = F.interpolate(self.compression5(x), size=[height_output, width_output], mode=inter_mode)
        x = self.relu(x + self.down5(x_))
        x_ = self.relu(x_ + middle)

        middle = F.interpolate(self.compression5_res(res), size=[height_output, width_output], mode=inter_mode)
        res = self.relu(res + self.down5_res(res_))
        res_ = self.relu(res_ + middle)

        # 1/32 1/8 -> x x_
        x = self.s61(x, res)
        x = self.spp(x)
        x_ = self.s61_(x_, res_)
        # concat
        x = x_ + F.interpolate(x, size=[height_output, width_output], mode=inter_mode)
        # x = self.concact(x_, x)

        logits = self.seghead(self.relu(self.bn(x)))
        if self.aux_mode == 'train':
            aux = self.aux_loss(aux)
            return logits, aux
        elif self.aux_mode == 'eval':
            return logits,
        elif self.aux_mode == 'pred':
            pred = logits.argmax(dim=1)
            return pred
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.load_pretrain()

    def load_pretrain(self):
        state = modelzoo.load_url(backbone_url)
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
        x = torch.randn(2, 3, 1024, 2048).cuda()
        model = Liner_Risidual(n_classes=19).cuda()
        outs = model(x)
        for out in outs:
            print(out.size())