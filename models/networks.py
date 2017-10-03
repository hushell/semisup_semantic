import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False,
             gpu_ids=[], last_layer='softmax'):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'st_resnet_9blocks' or which_model_netG == 'resnet_softmax_9blocks':
        from .style_transform_resnet import StyleTransformResNet
        netG = StyleTransformResNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                    gpu_ids=gpu_ids, last_layer=last_layer)
    #elif which_model_netG == 'unet_128':
    #    from models.u_net import UnetGenerator
    #    netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    #elif which_model_netG == 'unet_256':
    #    from models.u_net import UnetGenerator
    #    netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    #elif which_model_netG == 'resnet50_fcn':
    #    from models.resnet50_fcn import ResNet50FCN
    #    netG = ResNet50FCN(output_nc, freeze_batch_norm=False, gpu_ids=gpu_ids)
    elif which_model_netG == 'tiramisu_103':
        from .fc_densenet_tiramisu import FCDenseNet103
        netG = FCDenseNet103(output_nc, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)

    if use_gpu:
        netG.cuda(gpu_ids[0])
    if which_model_netG != 'resnet50_fcn':
        netG.apply(weights_init)

    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', gan_type='ls', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, gan_type=gan_type, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, gan_type=gan_type, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Discriminators
##############################################################################

# JSDGAN, LSGAN use PatchGAN discriminator
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, gan_type='ls', gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gan_type = gan_type
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if self.gan_type is 'jsd':
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)

        #if self.gan_type is 'wass':
        #    output = output.mean(0).view(1)
        return output


##############################################################################
# Losses
##############################################################################

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
