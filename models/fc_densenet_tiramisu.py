# **FirstConvLayer**
#
# * 3x3 Conv2D (pad=, stride=, in_chans=3, out_chans=48)
#
# **DenseLayer**
#
# * BatchNorm
# * ReLU
# * 3x3 Conv2d (pad=, stride=, in_chans=, out_chans=) - "no resolution loss" - padding included
# * Dropout (.2)
#
# **DenseBlock**
#
# * Input = FirstConvLayer, TransitionDown, or TransitionUp
# * Loop to create L DenseLayers (L=n_layers)
# * On TransitionDown we Concat(Input, FinalDenseLayerActivation)
# * On TransitionUp we do not Concat with input, instead pass FinalDenseLayerActivation to TransitionUp block
#
# **TransitionDown**
#
# * BatchNorm
# * ReLU
# * 1x1 Conv2D (pad=, stride=, in_chans=, out_chans=)
# * Dropout (0.2)
# * 2x2 MaxPooling
#
# **Bottleneck**
#
# * DenseBlock (15 layers)
#
# **TransitionUp**
#
# * 3x3 Transposed Convolution (pad=, stride=2, in_chans=, out_chans=)
# * Concat(PreviousDenseBlock, SkipConnection) - from cooresponding DenseBlock on transition down
#
# **FinalBlock**
#
# * 1x1 Conv2d (pad=, stride=, in_chans=256, out_chans=n_classes)
# * Softmax
#
# **FCDenseNet103 Architecture**
#
# * input (in_chans=3 for RGB)
# * 3x3 ConvLayer (out_chans=48)
# * DB (4 layers) + TD
# * DB (5 layers) + TD
# * DB (7 layers) + TD
# * DB (10 layers) + TD
# * DB (12 layers) + TD
# * Bottleneck (15 layers)
# * TU + DB (12 layers)
# * TU + DB (10 layers)
# * TU + DB (7 layers)
# * TU + DB (5 layers)
# * TU + DB (4 layers)
# * 1x1 ConvLayer (out_chans=n_classes) n_classes=11 for CamVid
# * Softmax
#
# **FCDenseNet67**
#
# * GrowthRate (k) = 16
# * 5 layers per dense block
# * 1 Conv Layer
# * 5 DenseBlocks Downsample (25 layers)
# * 5 TransitionDown
# * 5 Bottleneck layers
# * 5 Dense Blocks Upsample (25 layers)
# * 5 TransitionUp
# * 1 Conv Layer
# * 1 Softmax layer (doesn't count)
# 67 Total layers
#
# **360x480 Input Path**
#
# Image dimensions that are evenly divisible are nice. The 224x224 input work nicely w/out cropping.
# * skipsize torch.Size([1, 128, 360, 480])
# * skipsize torch.Size([1, 208, 180, 240])
# * skipsize torch.Size([1, 288, 90, 120])
# * skipsize torch.Size([1, 368, 45, 60])
# * skipsize torch.Size([1, 448, 22, 30])    <------- we lose 1 pixel here 22.5 to 22 b/c of rounding
# * bnecksize torch.Size([1, 80, 11, 15])
# * insize torch.Size([1, 80, 11, 15])
# * outsize torch.Size([1, 80, 22, 30])
# * insize torch.Size([1, 80, 22, 30])  <--------- we need to crop/pad to recover that lost pixel
# * outsize torch.Size([1, 80, 45, 60])
# * insize torch.Size([1, 80, 45, 60])
# * outsize torch.Size([1, 80, 90, 120])
# * insize torch.Size([1, 80, 90, 120])
# * outsize torch.Size([1, 80, 180, 240])
# * insize torch.Size([1, 80, 180, 240])
# * outsize torch.Size([1, 80, 360, 480])
#
#
# **224x224 Input Path**
# * skipsize torch.Size([3, 128, 224, 224])
# * skipsize torch.Size([3, 208, 112, 112])
# * skipsize torch.Size([3, 288, 56, 56])
# * skipsize torch.Size([3, 368, 28, 28])
# * skipsize torch.Size([3, 448, 14, 14])
# * bnecksize torch.Size([3, 80, 7, 7])
# * insize torch.Size([3, 80, 7, 7])
# * outsize torch.Size([3, 80, 14, 14])
# * insize torch.Size([3, 80, 14, 14])
# * outsize torch.Size([3, 80, 28, 28])
# * insize torch.Size([3, 80, 28, 28])
# * outsize torch.Size([3, 80, 56, 56])
# * insize torch.Size([3, 80, 56, 56])
# * outsize torch.Size([3, 80, 112, 112])
# * insize torch.Size([3, 80, 112, 112])
# * outsize torch.Size([3, 80, 224, 224])


import torch
import torch.nn as nn
import torch.nn.init as init

def center_crop(layer, max_height, max_width):
    #https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L162
    #Author does a center crop which crops both inputs (skip and upsample) to size of minimum dimension on both w/h
    batch_size, n_channels, layer_height, layer_width = layer.size()
    xy1 = (layer_width - max_width) // 2
    xy2 = (layer_height - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #author's impl - lasange 'same' pads with half
        # filter size (rounded down) on "both" sides
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
                out_channels=growth_rate, kernel_size=3, stride=1,
                  padding=1, bias=True))

        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super(DenseLayer, self).forward(x)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x

class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
              out_channels=in_channels, kernel_size=1, stride=1,
                padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super(TransitionDown, self).forward(x)

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
               out_channels=out_channels, kernel_size=3, stride=2,
              padding=0, bias=True) #crop = 'valid' means padding=0. Padding has reverse effect for transpose conv (reduces output size)
        #http://lasagne.readthedocs.io/en/latest/modules/layers/conv.html#lasagne.layers.TransposedConv2DLayer
        #self.updample2d = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out

class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super(Bottleneck, self).forward(x)


## Model
class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12,
                 gpu_ids=[]):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.gpu_ids = gpu_ids

        cur_channels_count = 0
        skip_connection_channel_counts = []

        #####################
        # First Convolution #
        #####################
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################
        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        #One final dense block
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        #####################
        #      Softmax      #
        #####################
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax()

        def f(x):
            #print("INPUT",x.size())
            out = self.firstconv(x)

            skip_connections = []
            for i in range(len(self.down_blocks)):
                #print("DBD size",out.size())
                out = self.denseBlocksDown[i](out)
                skip_connections.append(out)
                out = self.transDownBlocks[i](out)

            out = self.bottleneck(out)
            #print ("bnecksize",out.size())
            for i in range(len(self.up_blocks)):
                skip = skip_connections.pop()
                #print("DOWN_SKIP_PRE_UPSAMPLE",out.size(),skip.size())
                out = self.transUpBlocks[i](out, skip)
                #print("DOWN_SKIP_AFT_UPSAMPLE",out.size(),skip.size())
                out = self.denseBlocksUp[i](out)

            out = self.finalConv(out)
            out = self.softmax(out)
            return out

        self.model = f

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #kaiming is first name of author whose last name is 'He' lol
        init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def FCDenseNet57(n_classes, gpu_ids=[]):
    model = FCDenseNet(in_channels=3, down_blocks=(4, 4, 4, 4, 4),
                 up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
                 growth_rate=12, out_chans_first_conv=48, n_classes=n_classes, gpu_ids=gpu_ids)
    model.apply(weights_init)
    return model

def FCDenseNet67(n_classes, gpu_ids=[]):
    model = FCDenseNet(in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, gpu_ids=gpu_ids)
    model.apply(weights_init)
    return model

def FCDenseNet103(n_classes, gpu_ids=[]):
    model = FCDenseNet(in_channels=3, down_blocks=(4,5,7,10,12),
                 up_blocks=(12,10,7,5,4), bottleneck_layers=15,
                 growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, gpu_ids=gpu_ids)
    model.apply(weights_init)
    return model

