#!/usr/bin/python3
#coding=utf-8

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
        elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Unfold, nn.Sigmoid, nn.AdaptiveAvgPool2d,nn.Softmax,nn.Dropout2d)):
            pass
        else:
            m.initialize()
            

## Proposed Adjacent_Feature_Aligned_Module
"""
    Input: tell the input features's channel
"""
class AFAM(nn.Module):
    def __init__(self, in_channel = 128, out_channel = 64):
        super(AFAM, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.convA1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.bnA1   = nn.BatchNorm2d(self.in_channel)
        
        self.convB1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1)
        self.bnB1   = nn.BatchNorm2d(self.in_channel)

        self.convAB = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1)
        self.bnAB   = nn.BatchNorm2d(self.out_channel)

        self.delta_gen1 = nn.Sequential(
                        nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_channel),
                        nn.Conv2d(in_channel, 2, kernel_size=3, padding=1, bias=False)
                        )

        self.delta_gen2 = nn.Sequential(
                        nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_channel),
                        nn.Conv2d(in_channel, 2, kernel_size=3, padding=1, bias=False)
                        )


        self.delta_gen1[2].weight.data.zero_()
        self.delta_gen2[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[h/s, w/s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    
    """
    Input: x, y
    x is the low_stage's  feature map  [B, 128, H, W]
    y is the high_stage's feature map  [B, 128, H/2, W/2]
    """
    def forward(self, x, y):
        ## make y's shape the same as x
        y = F.interpolate(input=y, size=x.size()[2:], mode='bilinear', align_corners=True)
        ## By using muliply to  extract shared features from parent features through element-wise multiplication 
        fuze = torch.mul(x, y)
        ## Then add shared features back to parent features through element-wise addition to enhance them.
        y = F.relu(self.bnB1(self.convB1(fuze + y)), inplace=True)
        x = F.relu(self.bnA1(self.convA1(fuze + x)), inplace=True)
        ## merge the two processed features through the concatenation operation 
        concat = torch.cat((x, y), dim = 1)
        ##  by using an aligned feature aggregation function to enrich semantic information 
        ## get offset feature maps
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        ## Using offset feature maps into x, y
        y = self.bilinear_interpolate_torch_gridsample(y, x.size()[2:], delta1)
        x = self.bilinear_interpolate_torch_gridsample(x, x.size()[2:], delta2)
        ## Get final feature map
        out = x + y
        out = F.relu(self.bnAB(self.convAB(out)))
        return out
    
    def initialize(self):
        weight_init(self)