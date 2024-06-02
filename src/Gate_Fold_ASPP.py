import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU,nn.PReLU, nn.Unfold, nn.Sigmoid, nn.AdaptiveAvgPool2d,nn.Softmax,nn.Dropout2d)):
            pass
        else:
            m.initialize()
### Gate
class Gate(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Gate, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channels = self.in_channel, out_channels = self.out_channel, kernel_size = 3, padding = 1)
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = x * (self.Sigmoid(self.conv(x)))
        return y

    def initialize(self):
        weight_init(self)

###Gate-based Fold ASPP
class GFASPP(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 win_size=3, win_dilation=1, win_padding=0):
        super(GFASPP, self).__init__()
        #down_C = in_channel // 8
        self.down_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3,padding=1),nn.BatchNorm2d(out_channel),
             nn.PReLU())
        self.win_size = win_size
        self.unfold = nn.Unfold(win_size, win_dilation, win_padding, win_size)
        fold_C = out_channel * win_size * win_size
        down_dim = fold_C #// 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim,kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size, stride, padding, dilation, groups),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d( down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU() 
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, fold_C, kernel_size=1), nn.BatchNorm2d(fold_C), nn.PReLU()
        )

        # self.fold = nn.Fold(out_size, win_size, win_dilation, win_padding, win_size)
        self.gate1 = Gate(fold_C, fold_C)
        self.gate2 = Gate(fold_C, fold_C)
        self.gate3 = Gate(fold_C, fold_C)
        self.gate4 = Gate(fold_C, fold_C)

        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, in_feature):
        N, C, H, W = in_feature.size()
        in_feature = self.down_conv(in_feature)
        
        in_feature = self.unfold(in_feature)
        
        in_feature = in_feature.view(in_feature.size(0), in_feature.size(1),
                                     H // self.win_size, W // self.win_size)
        
        in_feature1 = self.conv1(in_feature)
        in_feature2 = self.conv2(in_feature + self.gate1(in_feature1))
        in_feature3 = self.conv3(in_feature + self.gate2(in_feature2))
        in_feature4 = self.conv4(in_feature + self.gate3(in_feature3))
        

        in_feature5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(in_feature + self.gate4(in_feature4), 1)), size=in_feature.size()[2:], mode='bilinear', align_corners=True)
        
        in_feature = self.fuse(torch.cat((in_feature1, in_feature2, in_feature3,in_feature4,in_feature5), 1))
        in_feature = in_feature.reshape(in_feature.size(0), in_feature.size(1), -1)


        in_feature = F.fold(input=in_feature, output_size=H, kernel_size=2, dilation=1, padding=0, stride=2)
        
        in_feature = self.up_conv(in_feature)
        
        return in_feature
    
    def initialize(self):
        weight_init(self)
