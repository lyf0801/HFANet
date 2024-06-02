

#!/usr/bin/python3
#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.PVT import pvt_v2_b1, PyramidVisionTransformerV2
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

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.inplanes = 64
        vgg = models.vgg16(pretrained=True)
        self.layer0 = nn.Sequential(*list(vgg.children())[0][0:5]) 
        self.layer1 = nn.Sequential(*list(vgg.children())[0][5:10]) 
        self.layer2 = nn.Sequential(*list(vgg.children())[0][10:17]) 
        self.layer3 = PyramidVisionTransformerV2(
        patch_size=2, in_chans = 256, embed_dims=[320], num_heads=[5], mlp_ratios=[4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2], sr_ratios=[1], num_stages = 1)
        self.layer4 = PyramidVisionTransformerV2(
        patch_size=2, in_chans = 320, embed_dims=[512], num_heads=[8], mlp_ratios=[4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2], sr_ratios=[1], num_stages = 1)

    def forward(self, x):
        out1 = self.layer0(x)
       
        out2 = self.layer1(out1)
        
        out3 = self.layer2(out2)
        
        out4 = self.layer3(out3)[0]  
        
        out5 = self.layer4(out4)[0]  
        
        return out1, out2, out3, out4, out5  
        
    def initialize(self):
        #self.load_state_dict(torch.load('./data/vgg16-397923af.pth'), strict=False)
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
        # Hybrid Backbone with PVT_b1 and VGG16
        self.bkbone   = VGG()
        # Fused Blocks for obtaining init feature maps
        self.fuse5 = AFAM(512, 64) 
        self.fuse4 = AFAM(320, 64)
        self.fuse3 = AFAM(256, 64)
        self.fuse2 = AFAM(128, 64)
        self.fuse1 = AFAM(64, 64)


        self.conv1 = nn.Conv2d(64, 320, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(320)

        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        # Gate_Flod_ASPP Module
        self.gate = nn.Sequential(GFASPP(in_channel=512,
                      out_channel=512,
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

    def forward(self, x):
        
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
