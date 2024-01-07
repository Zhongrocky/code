import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import init
from torch import transpose
from transunet import VisionTransformer
__all__ = ['VGGBlock', 'UNet']

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = DropBlock2D(block_size=7, keep_prob=0.9)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.dropout(out)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # # out = self.dropout(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        return out
class VGGBlock2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = DropBlock2D(block_size=7, keep_prob=0.9)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.dropout(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
# Unet
#
class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, is_train=True, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        # self.cpool_0 = CNN_pool(nb_filter[0], nb_filter[1])
        # self.cpool_1 = CNN_pool(nb_filter[1], nb_filter[2])
        # self.cpool_2 = CNN_pool(nb_filter[2], nb_filter[3])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.conv0_0 = VGGBlock2(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock2(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock2(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock2(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv2_1 = VGGBlock2(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock2(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock2(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        # pl_0 = self.pool(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        # x1_0 = self.conv1_0(self.cpool_0(x0_0))
        # x2_0 = self.conv2_0(self.cpool_1(x1_0))
        # x3_0 = self.conv3_0(self.cpool_2(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        output = self.final(x0_3)
        return output
class UNet_3(nn.Module):
    def __init__(self, num_classes, input_channels=3, is_train=True, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        # self.cpool_0 = CNN_pool(nb_filter[0], nb_filter[1])
        # self.cpool_1 = CNN_pool(nb_filter[1], nb_filter[2])
        # self.cpool_2 = CNN_pool(nb_filter[2], nb_filter[3])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.conv0_0 = VGGBlock2(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock2(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock2(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock2(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv2_1 = VGGBlock2(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock2(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock2(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        # pl_0 = self.pool(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        # x3_0 = self.conv3_0(self.pool(x2_0))
        # x1_0 = self.conv1_0(self.cpool_0(x0_0))
        # x2_0 = self.conv2_0(self.cpool_1(x1_0))
        # x3_0 = self.conv3_0(self.cpool_2(x2_0))
        # x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        output = self.final(x0_3)
        return output
class UNet_5(nn.Module):
    def __init__(self, num_classes, input_channels=3, is_train=True, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        # self.cpool_0 = CNN_pool(nb_filter[0], nb_filter[1])
        # self.cpool_1 = CNN_pool(nb_filter[1], nb_filter[2])
        # self.cpool_2 = CNN_pool(nb_filter[2], nb_filter[3])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.conv0_0 = VGGBlock2(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock2(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock2(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock2(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3 = VGGBlock2(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock2(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock2(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock2(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        # pl_0 = self.pool(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        # x1_0 = self.conv1_0(self.cpool_0(x0_0))
        # x2_0 = self.conv2_0(self.cpool_1(x1_0))
        # x3_0 = self.conv3_0(self.cpool_2(x2_0))
        x3 = self.conv3(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        output = self.final(x0_3)
        return output

###设计的最终模型
class MSFF_4_enatt_UNetresv4_sigv9(nn.Module):
    def __init__(self, num_classes, input_channels=3, is_train=True, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_first = nn.Dropout(p=0.025)
        self.drop = nn.Dropout(p=0.05)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.mff0 = MSFF_sigv9(input_channels, input_channels)
        self.mff1 = MSFF_sigv9(nb_filter[0], nb_filter[0])
        self.mff2 = MSFF_sigv9(nb_filter[1], nb_filter[1])
        self.mff3 = MSFF_sigv9(nb_filter[2], nb_filter[2])

        # self.conv2_1 = VGGBlock2(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        # self.conv1_2 = VGGBlock2(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        # self.conv0_3 = VGGBlock2(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv2_1 = ResDecoderv5(nb_filter[3] + nb_filter[2], nb_filter[2])
        self.conv1_2 = ResDecoderv5(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_3 = ResDecoderv5(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # x0_0 = self.conv0_0(self.mff0(input))
        # x1_0 = self.conv1_0(self.mff1(self.pool(x0_0)))
        # x2_0 = self.conv2_0(self.drop_first(self.mff2(self.pool(x1_0))))
        # x3_0 = self.conv3_0(self.drop(self.mff3(self.pool(x2_0))))
        #
        # x2_2 = self.conv2_1(self.drop(torch.cat([x2_0, self.up(x3_0)], 1)))
        # x1_3 = self.conv1_2(self.drop(torch.cat([x1_0, self.up(x2_2)], 1)))
        # x0_4 = self.conv0_3(self.drop(torch.cat([x0_0, self.up(x1_3)], 1)))
        #
        # output = self.final(x0_4)

        x0_0 = self.conv0_0(self.mff0(input))
        x1_0 = self.conv1_0(self.mff1(self.pool(x0_0)))
        x2_0 = self.conv2_0(self.mff2(self.pool(x1_0)))
        x3_0 = self.conv3_0(self.mff3(self.pool(x2_0)))

        x2_2 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_3 = self.conv1_2(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_3(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)

        return output

###消融实验
class MSFF_4_enatt_UNetnoresv4_sigv9(nn.Module):
    def __init__(self, num_classes, input_channels=3, is_train=True, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_first = nn.Dropout(p=0.025)
        self.drop = nn.Dropout(p=0.05)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.mff0 = MSFF_sigv9(input_channels, input_channels)
        self.mff1 = MSFF_sigv9(nb_filter[0], nb_filter[0])
        self.mff2 = MSFF_sigv9(nb_filter[1], nb_filter[1])
        self.mff3 = MSFF_sigv9(nb_filter[2], nb_filter[2])

        self.conv2_1 = VGGBlock2(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock2(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock2(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        # self.conv2_1 = ResDecoderv5(nb_filter[3] + nb_filter[2], nb_filter[2])
        # self.conv1_2 = ResDecoderv5(nb_filter[1] + nb_filter[2], nb_filter[1])
        # self.conv0_3 = ResDecoderv5(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(self.mff0(input))
        x1_0 = self.conv1_0(self.mff1(self.pool(x0_0)))
        x2_0 = self.conv2_0(self.drop_first(self.mff2(self.pool(x1_0))))
        x3_0 = self.conv3_0(self.drop(self.mff3(self.pool(x2_0))))

        x2_2 = self.conv2_1(self.drop(torch.cat([x2_0, self.up(x3_0)], 1)))
        x1_3 = self.conv1_2(self.drop(torch.cat([x1_0, self.up(x2_2)], 1)))
        x0_4 = self.conv0_3(self.drop(torch.cat([x0_0, self.up(x1_3)], 1)))

        output = self.final(x0_4)

        return output
class MSFF_4_enatt_UNetresv4_nosigv9(nn.Module):
    def __init__(self, num_classes, input_channels=3, is_train=True, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_first = nn.Dropout(p=0.025)
        self.drop = nn.Dropout(p=0.05)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.conv0_0 = VGGBlock2(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock2(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock2(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock2(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.conv2_1 = VGGBlock2(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        # self.conv1_2 = VGGBlock2(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        # self.conv0_3 = VGGBlock2(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv2_1 = ResDecoderv5(nb_filter[3] + nb_filter[2], nb_filter[2])
        self.conv1_2 = ResDecoderv5(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_3 = ResDecoderv5(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.drop_first(self.pool(x1_0)))
        x3_0 = self.conv3_0(self.drop(self.pool(x2_0)))

        x2_2 = self.conv2_1(self.drop(torch.cat([x2_0, self.up(x3_0)], 1)))
        x1_3 = self.conv1_2(self.drop(torch.cat([x1_0, self.up(x2_2)], 1)))
        x0_4 = self.conv0_3(self.drop(torch.cat([x0_0, self.up(x1_3)], 1)))

        output = self.final(x0_4)

        return output

###FFNET
'''
FFNet depth = 4
'''
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class ASPP(nn.Module):
    def __init__(self, inplanes=1024, mid_c=256, dilations=[1, 6, 12, 18], factor=1):
        super(ASPP, self).__init__()
        # self.conv0 = nn.Conv2d(inplanes , inplanes // factor, 1, bias=False)
        self.aspp1 = _ASPPModule(inplanes // factor, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes // factor, mid_c, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes // factor, mid_c, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes // factor, mid_c, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes // factor, mid_c, 1, stride=1, bias=False),
                                             # nn.BatchNorm2d(mid_c),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(mid_c * 5, inplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        # x = self.conv0(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
class resconv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block, self).__init__()
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
class FFNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(FFNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = single_conv(ch_in=input_channels, ch_out=64)
        self.Conv2 = resconv_block(ch_in=64, ch_out=128)
        self.Conv3 = resconv_block(ch_in=128, ch_out=256)
        self.Conv4 = resconv_block(ch_in=256, ch_out=512)

        self.center = ASPP(512, 256)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv1_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv2_1x1 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv3_1x1 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv5_1x1 = nn.Conv2d(3 * num_classes, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x4 = self.center(x4)

        # decoding + concat path

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        p1 = self.Conv1_1x1(d2)
        p2 = F.interpolate(self.Conv2_1x1(d3), size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.Conv3_1x1(d4), size=p1.shape[2:], mode='bilinear', align_corners=False)

        p = torch.cat((p1, p2, p3), 1)
        p = self.Conv5_1x1(p)

        return p

####residual unet
class SepConv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        #self.dropout = DropBlock2D(block_size=7, keep_prob=0.9)
        self.con1x1 = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out1 = self.con1x1(x)
        out2 = self.bn1(x)
        #out = self.dropout(out)
        out2 = self.relu(out2)
        out2 = self.conv1(out2)

        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        # out = self.dropout(out)
        out = out1 + out2
        return out
class cnn1x1(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(cnn1x1, self).__init__()

        self.con1x1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1)

    def forward(self, input):
        return self.con1x1(input)
class Res_unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, is_train=True, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 96, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        # self.cpool_0 = CNN_pool(nb_filter[0], nb_filter[1])
        # self.cpool_1 = CNN_pool(nb_filter[1], nb_filter[2])
        # self.cpool_2 = CNN_pool(nb_filter[2], nb_filter[3])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.conv0_0 = VGGBlock2(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SepConv(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SepConv(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SepConv(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SepConv(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv5_0 = SepConv(nb_filter[4], nb_filter[5], nb_filter[5])
        self.drop = nn.Dropout(p=0.05)
        self.conv1x1 = cnn1x1(nb_filter[5], nb_filter[5])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv5_1 = SepConv(nb_filter[5] + nb_filter[5], nb_filter[5], nb_filter[5])
        self.conv4_2 = SepConv(nb_filter[5] + nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv3_3 = SepConv(nb_filter[4] + nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv2_4 = SepConv(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_5 = SepConv(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_6 = SepConv(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        # pl_0 = self.pool(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x5_0 = self.conv5_0(self.pool(x4_0))
        x5_0_drop = self.conv1x1(self.drop(x5_0))

        # x1_0 = self.conv1_0(self.cpool_0(x0_0))
        # x2_0 = self.conv2_0(self.cpool_1(x1_0))
        # x3_0 = self.conv3_0(self.cpool_2(x2_0))
        x5_1 = self.conv5_1(torch.cat([x5_0, x5_0_drop], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, self.up(x4_2)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_0, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(torch.cat([x0_0, self.up(x1_5)], 1))
        output = self.final(x0_6)
        return output

class ResDecoderv4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDecoderv4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv1x1 = FSM(in_channels, out_channels)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        # out = self.relu(out)
        return out


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)

class Aup1(nn.Module):
    def __init__(self, Maxpool, up, int_channel):
        super(Aup1, self).__init__()
        self.W_p1 = nn.Sequential(
            nn.Conv2d(Maxpool, int_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channel)
        )
        self.W_p2 = nn.Sequential(
            nn.Conv2d(up, int_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channel)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(int_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.psi1 = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.psi2 = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pool, up, x):
        Pool = self.W_p1(pool)
        Up = self.W_p2(up)
        x1 = self.W_x(x)
        add1 = torch.add(Pool, x1)
        add2 = torch.add(Up, x1)
        act_1 = self.relu(add1)
        act_2 = self.relu(add2)
        psi_1 = self.psi1(act_1)
        psi_2 = self.psi2(act_2)
        cat = torch.cat([psi_1, psi_2], 1)
        psi_ = self.psi(cat)
        return x * psi_
class Aup2(nn.Module):
    def __init__(self, Maxpool, up, int_channel):
        super(Aup2, self).__init__()
        self.W_p1 = nn.Sequential(
            nn.Conv2d(Maxpool, int_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channel)
        )
        self.W_p2 = nn.Sequential(
            nn.Conv2d(up, int_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int_channel)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(int_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.psi1 = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.psi2 = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pool, up, x):
        Pool = self.W_p1(pool)
        Up = self.W_p2(up)
        x1 = self.W_x(x)
        add1 = torch.add(Pool, x1)
        add2 = torch.add(Up, x1)
        act_1 = self.relu(add1)
        act_2 = self.relu(add2)
        psi_1 = self.psi1(act_1)
        psi_2 = self.psi2(act_2)
        cat = torch.cat([psi_1, psi_2], 1)
        psi_ = self.psi(cat)
        return x * psi_
class MDACM_Unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.relu = nn.ReLU(inplace=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.drop_first = nn.Dropout(p=0.025)
        self.drop = nn.Dropout(p=0.05)
        self.AG1020 = Aup1(nb_filter[1], nb_filter[3], nb_filter[2])
        self.Ag0010 = Aup2(nb_filter[0], nb_filter[2], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])

        self.conv1_2 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # 测试
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.drop_first(self.pool(x0_0)))
        x2_0 = self.conv2_0(self.drop(self.pool(x1_0)))
        x3_0 = self.conv3_0(self.drop(self.pool(x2_0)))

        # x0_0 = self.conv0_0(self.mff0(input))
        # x1_0 = self.conv1_0(self.drop_first(self.mff1(self.pool(x0_0))))
        # x2_0 = self.conv2_0(self.drop(self.mff2(self.pool(x1_0))))
        # x3_0 = self.conv3_0(self.drop(self.mff3(self.pool(x2_0))))

        ag1020 = self.AG1020(self.pool(x1_0), self.up(x3_0), x2_0)
        x2_1 = self.conv2_1(self.drop(torch.cat([ag1020, self.up(x3_0)], 1)))
        ag0010 = self.Ag0010(self.pool(x0_0), self.up(x2_1), x1_0)
        x1_2 = self.conv1_2(self.drop(torch.cat([ag0010, self.up(x2_1)], 1)))
        x0_3 = self.conv0_3(self.drop(torch.cat([x0_0, self.up(x1_2)], 1)))
        out = self.final(x0_3)
        return out
        # 训练
        # x0_0 = self.conv0_0(input)
        #
        # x1_0 = self.conv1_0(self.pool(x0_0))
        # x2_0 = self.conv2_0(self.pool(x1_0))
        # x3_0 = self.conv3_0(self.pool(x2_0))
        #
        # ag1020 = self.AG1020(self.pool(x1_0), self.up(x3_0), x2_0)
        # x2_1 = self.conv2_1(torch.cat([ag1020, self.up(self.drop1(x3_0))], 1))
        #
        # ag0010 = self.Ag0010(self.pool(x0_0), self.up(x2_1), x1_0)
        # x1_2 = self.conv1_2(torch.cat([ag0010, self.up(self.drop2(x2_1))], 1))
        #
        # x0_3 = self.conv0_3(torch.cat([x0_0, self.up(self.drop3(x1_2))], 1))
        #
        # out = self.final(x0_3)
        #
        # return out
class Attention_Unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.relu = nn.ReLU(inplace=True)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        self.AG1020 = ABD3377(nb_filter[1], nb_filter[2], nb_filter[2])
        self.Ag0010 = ABD3377(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_1 = VGGBlock(nb_filter[3] + nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        poolx1_0 = self.pool(x1_0)
        ag1020 = self.AG1020(pool=poolx1_0, x=x2_0)
        x2_1 = self.conv2_1(torch.cat([ag1020, self.up(x3_0)], 1))

        poolx0_0 = self.pool(x0_0)

        ag0010 = self.Ag0010(pool=poolx0_0, x=x1_0)
        x1_2 = self.conv1_2(torch.cat([ag0010, self.up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        output = self.final(x0_3)

        return output
class ABD3377(nn.Module):
    def __init__(self, pool_channel, x_channel, int_channel):
        super(ABD3377, self).__init__()
        self.W_p = nn.Sequential(
            nn.Conv2d(pool_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channel, int_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channel)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_channel, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pool, x):
        pool1 = self.W_p(pool)
        x1 = self.W_x(x)
        psi = self.relu(torch.add(pool1, x1))
        psi_ = self.psi(psi)
        return x * psi_

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    # model = VisionTransformer(input_channels=1, num_classes=1).to(device)
    model = VisionTransformer(img_size=512, num_classes=1).to(device)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print("模型参数数量：", model_size_mb)
    # summary(model, (1, 512, 512))
    x = torch.rand(2, 1, 512, 512)
    x = x.to(device)
    print(model(x).size())
    # y= model.forward(x)
    # print()
    # dummy_input01 = torch.rand(2, 1, 512, 512).to(device)  # 假设输入10张1*28*28的图片
    # with SummaryWriter(comment='Net') as w:
    #     w.add_graph(model, (dummy_input01))


if __name__ == '__main__':
    main()