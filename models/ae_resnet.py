import torch
import torch.nn as nn
import numpy as np

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class AEResNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, last_layer='softmax',
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        assert(n_blocks >= 0)
        super(AEResNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # Encoder
        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3), # out_size = input_size
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1), # out_size = (input_size - 1) / 2 + 1
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks): # out_size = input_size; out_nc = input_nc
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]
        self.encoder = nn.Sequential(*model)

        # Decoder
        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1), # out_size = input_size * 2
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]
        self.decoder = nn.Sequential(*model)

        # Classifier
        model = [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)] # out_size = input_size

        if last_layer == 'softmax':
            model += [nn.LogSoftmax(dim=1)]
        elif last_layer == 'tanh':
            model += [nn.Tanh()]
        else:
            print('Layer name [%s] is not recognized' % last_layer)
        self.classifier = nn.Sequential(*model)


    def forward(self, input):
        z = self.encoder(input)
        z = self.decoder(z)
        return self.classifier(z), z


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
