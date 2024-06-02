#!/usr/bin/python3
#coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


## salient maps: BCEWithLogs + wIOU
def structure_loss(preds, prede, masks):
  
    wbce  = F.binary_cross_entropy_with_logits(preds, masks)
    wbce = wbce * (torch.sigmoid(prede) * 4 + 1)  #edge guided 
    preds  = torch.sigmoid(preds)
    inter = (preds*masks).sum(dim=(2,3))
    union = (preds+masks).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return wbce.mean()+wiou.mean()

## edges maps： BCEWithLogs
def criterion(prede, preds, maske):

    wbce =  F.binary_cross_entropy_with_logits(prede, maske).cuda()  
    wbce = wbce * ((1- torch.sigmoid(preds) * maske) * 4 + 1)  #saliency guided
    return wbce.mean()
