

#!/usr/bin/python3
#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.PVT import pvt_v2_b2, PyramidVisionTransformerV2
from functools import partial
from src.Gate_Fold_ASPP import GFASPP
from src.AFAM import AFAM




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

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        #self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer3 = PyramidVisionTransformerV2(
        patch_size=2, in_chans = 512, embed_dims=[320], num_heads=[5], mlp_ratios=[4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[6], sr_ratios=[2], num_stages = 1)
        #self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.layer4 = PyramidVisionTransformerV2(
        patch_size=2, in_chans = 320, embed_dims=[512], num_heads=[8], mlp_ratios=[4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3], sr_ratios=[1], num_stages = 1)
        
    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        
        out2 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out2)
        
        out3 = self.layer2(out2)
        
        out4 = self.layer3(out3)[0] 
        
        out5 = self.layer4(out4)[0]  
        
       
        return out1, out2, out3, out4, out5 

    def initialize(self):
        #self.load_state_dict(torch.load('./data/resnet50-19c8e357.pth'), strict=False)
        pass



class PredictBlock(nn.Module):
    """
    Input: 
           Type: Tensor 
           Note: feature maps after AFAM for Deep Supervised
           Channel: 64
           Size: B * 64 * 448 * 448
    Output:
           Type: Tensor list
           Len: 2
           Note: salient maps & edge maps
           Channel: 1
           Size: B * 1 * 448 * 448
    """
    def __init__(self):
        super(PredictBlock, self).__init__()
        self.down1 = nn.Conv2d(64, 32, kernel_size=3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.smaps = nn.Conv2d(32, 1, kernel_size = 3, padding = 1)
        self.edges = nn.Conv2d(32, 1, kernel_size = 3, padding = 1)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.down1(x)))
        smaps = self.smaps(out)
        edges = self.edges(out)
        return  smaps, edges
        
    def initialize(self):
        weight_init(self)

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # Hybrid Backbone with PVT and ResNet50
        self.bkbone   = ResNet()
        # Fused Blocks for obtaining init feature maps
        self.fuse5 = AFAM(512, 64) 
        self.fuse4 = AFAM(320, 64)
        self.fuse3 = AFAM(512, 64)
        self.fuse2 = AFAM(256, 64)
        self.fuse1 = AFAM(64, 64)

        self.conv1 = nn.Conv2d(64, 320, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(320)

        self.conv2 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        # Gate_Flod_ASPP Module
        self.gate = nn.Sequential(GFASPP(in_channel=512,
                      out_channel=512 , 
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2,
                      win_size=2,
                      win_padding=0,
        ), nn.BatchNorm2d(512), nn.PReLU())  

        #Prediction Module
      
        self.prerdict5 = PredictBlock()
        self.prerdict4 = PredictBlock()
        self.prerdict3 = PredictBlock()
        self.prerdict2 = PredictBlock()
        self.prerdict1 = PredictBlock()

        self.initialize()

    def forward(self, x, shape=None):

        
        s1, s2, s3, s4, s5 = self.bkbone(x) 
        
        s6 = self.gate(s5)  
        
        
        out5 = self.fuse5(s6, s5)     
       
        out4 = self.fuse4(s4, F.relu(self.bn1(self.conv1(out5))))
        out3 = self.fuse3(s3, F.relu(self.bn2(self.conv2(out4))))
        out2 = self.fuse2(s2, F.relu(self.bn3(self.conv3(out3))))
        out1 = self.fuse1(s1, F.relu(self.bn4(self.conv4(out2))))
        
        
        if self.training:  
            
            smap1, edge1 = self.prerdict1(out1)
            smap2, edge2 = self.prerdict2(out2)
            smap3, edge3 = self.prerdict3(out3)
            smap4, edge4 = self.prerdict4(out4)
            smap5, edge5 = self.prerdict5(out5)
            
            smap1 = F.interpolate(smap1, size = x.size()[2:], mode='bilinear',align_corners=True)
            smap2 = F.interpolate(smap2, size = x.size()[2:], mode='bilinear',align_corners=True)
            smap3 = F.interpolate(smap3, size = x.size()[2:], mode='bilinear',align_corners=True)
            smap4 = F.interpolate(smap4, size = x.size()[2:], mode='bilinear',align_corners=True)
            smap5 = F.interpolate(smap5, size = x.size()[2:], mode='bilinear',align_corners=True)
           
            edge1 = F.interpolate(edge1, size = x.size()[2:], mode='bilinear',align_corners=True)
            edge2 = F.interpolate(edge2, size = x.size()[2:], mode='bilinear',align_corners=True)
            edge3 = F.interpolate(edge3, size = x.size()[2:], mode='bilinear',align_corners=True)
            edge4 = F.interpolate(edge4, size = x.size()[2:], mode='bilinear',align_corners=True)
            edge5 = F.interpolate(edge5, size = x.size()[2:], mode='bilinear',align_corners=True)
            
            return smap1, smap2, smap3, smap4, smap5,   edge1, edge2, edge3, edge4, edge5
            
        else:
            smap1, edge1 = self.prerdict1(out1)
            smap1 = F.interpolate(smap1, size = x.size()[2:], mode='bilinear',align_corners=True)
            return torch.sigmoid(smap1)


    def initialize(self):
        weight_init(self)

