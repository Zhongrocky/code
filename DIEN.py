import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import init
from torch import transpose



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
class ResDecoderv5(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ResDecoderv5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        # self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1 = FSM(in_channels, out_channels)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        # out = self.relu(out)
        return out
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        #self.dropout = DropBlock2D(block_size=7, keep_prob=0.9)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
    def forward(self, x):
        out = self.conv1(x)
        #out = self.dropout(out)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # # out = self.dropout(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        return out
class MSFF_sigv9(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # self.pooling_r = 4
        self.con1x1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1)
        self.con_eca = nn.Conv2d(in_channels=input_channel * 4, out_channels=output_channel, kernel_size=1, stride=1)
        self.con3x3 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1,
                                padding=1)
        self.ccon1x1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1,
                      ),
            nn.BatchNorm2d(output_channel),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel*2, kernel_size=3, stride=1,
                      ),
            nn.BatchNorm2d(output_channel*2),
            nn.Conv2d(input_channel*2, output_channel, kernel_size=1, stride=1,padding=1
                      ),
            # nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )
    def forward(self, input):
        x = self.con1x1(input)
        x1 = x

        x2 =self.con3x3(input)
        x3 = self.con3x3(F.interpolate(self.k2(input+x2), input.size()[2:]))
        x4 = self.con3x3(self.k3(input + x3))
        # fusion = self.con1x1(x1 + x2 + x3 + x4)
        fusion = self.con_eca(torch.cat([x1, x2, x3, x4], dim=1))
        # s1 = self.con1x1(x1 + fusion)
        # s2 = self.con1x1(x2 + fusion)
        # s3 = self.con1x1(x3 + fusion)
        # s4 = self.con1x1(x4 + fusion)
        # feature = self.ccon1x1(torch.cat([s1, s2, s3, s4], dim=1))
        out = input + input * self.sig(fusion)
        return out
class DIEN(nn.Module):
    def __init__(self, num_classes, input_channels=3,  is_train=True, deep_supervision = False):
        super().__init__()

        nb_filter = [32, 64, 128, 256]
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_first = nn.Dropout(p=0.025)
        self.drop = nn.Dropout(p=0.05)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值
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
        #self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

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
