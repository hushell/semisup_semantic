# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:12:44 2017

@author: spyros
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.models
import numpy as np

def init_parameters(mult):
    def init_parameters_(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            fin = mult * np.prod(m.kernel_size) * m.in_channels
            std_val = np.sqrt(2.0/fin)
            m.weight.data.normal_(0.0, std_val)
            m.bias.data.fill_(0.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    return init_parameters_

def freeze_batch_norm_fun(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.training = False
        m.running_mean.requires_grad = False
        m.running_var.requires_grad = False
        if m.affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False

class ResNet50FCN(nn.Module):
    def __init__(self, num_out_channels, freeze_batch_norm=True, gpu_ids=[]):
        super(ResNet50FCN, self).__init__()

        self.num_out_channels = num_out_channels
        self.freeze_batch_norm = freeze_batch_norm
        self.gpu_ids = gpu_ids

        resnet = torchvision.models.resnet50(pretrained=True)

        # feature blocks with pretrained weights
        self.feat_block0 = nn.Sequential(
                                resnet.conv1,
                                resnet.bn1,
                                resnet.relu)
        self.feat_block1 = nn.Sequential(
                                resnet.maxpool,
                                resnet.layer1)
        self.feat_block2 = resnet.layer2
        self.feat_block3 = resnet.layer3
        self.feat_block4 = resnet.layer4

        if self.freeze_batch_norm:
            self.feat_block0.apply(freeze_batch_norm_fun)
            self.feat_block1.apply(freeze_batch_norm_fun)
            self.feat_block2.apply(freeze_batch_norm_fun)
            self.feat_block3.apply(freeze_batch_norm_fun)
            self.feat_block4.apply(freeze_batch_norm_fun)

        # prediction blocks
        self.pred_block4 = nn.Sequential(
                                nn.Conv2d(2048, 512, 3, stride=1, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, self.num_out_channels, 7, stride=1, padding=3, bias=True),
                                nn.UpsamplingBilinear2d(scale_factor=8))
        self.pred_block4.apply(init_parameters(1.0))

        self.pred_block3 = nn.Sequential(
                                nn.Conv2d(1024, 256, 3, stride=1, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, self.num_out_channels, 5, stride=1, padding=2, bias=True),
                                nn.UpsamplingBilinear2d(scale_factor=4))
        self.pred_block3.apply(init_parameters(2.0))

        self.pred_block2 = nn.Sequential(
                                nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, self.num_out_channels, 5, stride=1, padding=2, bias=True),
                                nn.UpsamplingBilinear2d(scale_factor=2))
        self.pred_block2.apply(init_parameters(1.0))

        self.pred_block1 = nn.Sequential(
                                nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, self.num_out_channels, 5, stride=1, padding=2, bias=True))
        self.pred_block1.apply(init_parameters(0.5))

        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=4)
        self.log_softmax = nn.LogSoftmax()

        def f(input):
            feat0 = self.feat_block0(input)
            feat1 = self.feat_block1(feat0)
            feat2 = self.feat_block2(feat1)
            feat3 = self.feat_block3(feat2)
            feat4 = self.feat_block4(feat3)

            out4  = self.pred_block4(feat4)
            out3  = self.pred_block3(feat3)
            out2  = self.pred_block2(feat2)
            out1  = self.pred_block1(feat1)
            ave_out = out4+out3+out2+out1

            output = self.final_upsample(ave_out)
            output = self.log_softmax(output)
            return output
        self.model = f

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


if __name__ == '__main__':
    num_out_channels = 20
    freeze_batch_norm = True
    network = ResNet50FCN(num_out_channels, freeze_batch_norm)
