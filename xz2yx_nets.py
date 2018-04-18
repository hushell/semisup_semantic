import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
import argparse

#########################################################################
# OPTIONS
parser = argparse.ArgumentParser()
################################
# optimizer
################################
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

################################
# data settings
################################
parser.add_argument('--dataset', type=str, default='cityscapesAB', help='chooses which dataset is loaded. [cityscapesAB | pascal | camvid]')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--resize_or_crop', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|no_resize]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
parser.add_argument('--ignore_index', type=int, default=-100, help='mask this class without contributing to nll_loss')
parser.add_argument('--unsup_portion', type=int, default=9, help='portion of unsupervised, range=0,...,10')
parser.add_argument('--portion_total', type=int, default=10, help='total portion of unsupervised, e.g., 10')
parser.add_argument('--unsup_sampler', type=str, default='sep', help='unif, sep, unif_ignore')

################################
# train settings
################################
parser.add_argument('--name', type=str, default='xyx', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--checkpoints_dir', default='ckpt', help='folder to output images and model checkpoints')
parser.add_argument('--save_every', default=2, type=int, help='')
parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')
parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--gpu_ids', type=str, default='', help='gpu ids: e.g. 0; 0,2')
parser.add_argument('--stage', type=str, default='X2Z:2,Z2Y:2,ZY2X:0,D:0', help='2 train, 1 free, 0 disable')
parser.add_argument('--lrs', type=str, default='1e-4,1e-4,1e-4,1e-4', help='X2Z,Z2Y,ZY2X,D')
parser.add_argument('--lambdas', type=str, default='1e-0,1e-0', help='lambda_x, lambda_y')

################################
# model settings
################################
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--widthSize', type=int, default=256, help='crop to this width')
parser.add_argument('--heightSize', type=int, default=256, help='crop to this height')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=20, help='# of output image channels')
parser.add_argument('--z_nc', type=int, default=40, help='# of output image channels')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--noise', default='sphere', help='normal|sphere')
parser.add_argument('--n_layers_D', type=int, default=3, help='')
parser.add_argument('--x_drop', type=float, default=0.9, help='x dropout rate')

################################
# external
################################
parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--port', type=int, default=8097, help='port of visdom')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')


#########################################################################
def populate_xy(x, y_int, dataloader, opt):
    AtoB = opt.which_direction == 'AtoB'
    real_cpu = dataloader.next()
    if x is not None:
        x_cpu = real_cpu['A' if AtoB else 'B']
        #x.data.resize_(x_cpu.size()).copy_(x_cpu)
        assert(x.size() == x_cpu.size())
        x.copy_(x_cpu)
    if y_int is not None:
        y_cpu = real_cpu['B' if AtoB else 'A']
        assert(y_int.size() == y_cpu.size())
        y_int.copy_(y_cpu)
    #if opt.DEBUG:
    #    print(real_cpu['A_paths'])

def one_hot(y_int, opt):
    ''' y_int is a Variable '''
    y_temp = y_int.unsqueeze(dim=1)
    y = torch.FloatTensor(y_int.size(0), opt.output_nc, y_int.size(1), y_int.size(2))
    if len(opt.gpu_ids) > 0:
        y = y.cuda()
    y.zero_().scatter_(1, y_temp, 1)
    return y

def gumbel_softmax(log_probs, temperature=1.0, gpu_ids=[], eps=1e-20):
    ''' sample a multinomial RV by one_hot(argmax_y [log_prob_y + g_y])
        where argmax is approx by softmax(log_prob / tau), exact when tau -> 0
        NOTE: log_probs is a Variable
    '''
    if temperature == 0:
        return log_probs
    # Gumbel(0,1): -log( -log(U + eps) + eps )
    noise = torch.rand(log_probs.size())
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    if len(gpu_ids) > 0:
        noise = noise.cuda()
    noise = Variable(noise)
    return F.log_softmax( (log_probs + noise) / temperature, dim=1 )

def noise_log_y(y, temperature=1.0, gpu_ids=[], eps=1e-20):
    ''' log(y + eps) = log_prob, where y is one_hot
        return log_softmax( log_prob + gumbel(0,1) / tau ), which is a sample near GT
        NOTE: y is a Variable
    '''
    log_probs_gt = torch.log(y + eps)
    return gumbel_softmax(log_probs_gt, temperature, gpu_ids, eps) # add noise

def create_losses(opt):
    CE = torch.nn.NLLLoss2d(ignore_index=opt.ignore_index)
    L1 = torch.nn.L1Loss()
    if len(opt.gpu_ids) > 0:
        CE = CE.cuda(opt.gpu_ids[0])
        L1 = L1.cuda(opt.gpu_ids[0])
    return CE, L1

#########################################################################
from models import networks
from models.networks import weights_init

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# net X -> Z
class FX2Z(nn.Module):
    def __init__(self, opt, n_blocks=5):
        super(FX2Z, self).__init__()
        self.gpu_ids = opt.gpu_ids
        norm_layer=nn.BatchNorm2d
        use_bias = norm_layer == nn.InstanceNorm2d
        use_dropout = opt.use_dropout
        padding_type='reflect'
        ngf = opt.ngf
        input_nc = opt.input_nc
        z_nc = opt.z_nc
        n_downsampling = 2

        # init conv: input_nc -> ngf
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # downsampling x4: ngf -> ndf * 2^2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # resnet blocks: ngf * 4 -> ngf * 4
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # upsampling blocks: ngf*4 -> ngf
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        # final conv: ngf -> z_nc
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, z_nc, kernel_size=7, padding=0)]
        model += [nn.LogSoftmax(dim=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# net Z -> Y
class FZ2Y(nn.Module):
    def __init__(self, opt):
        super(FZ2Y, self).__init__()
        self.gpu_ids = opt.gpu_ids
        z_nc = opt.z_nc
        output_nc = opt.output_nc

        # z_nc -> y_nc by 1x1 conv
        model = []
        model += [nn.Conv2d(z_nc, output_nc, kernel_size=1, stride=1, padding=0)]
        model += [nn.LogSoftmax(dim=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# net Z -> X: G(z)
class GZ2X(nn.Module):
    def __init__(self, opt, n_blocks=5):
        super(GZ2X, self).__init__()
        self.gpu_ids = opt.gpu_ids
        self.x_drop = opt.x_drop
        norm_layer=nn.BatchNorm2d # TODO: try instanceNorm
        use_bias = norm_layer == nn.InstanceNorm2d
        use_dropout = opt.use_dropout
        padding_type='reflect'
        n_downsampling = 2
        ngf = opt.ngf
        input_nc = opt.input_nc
        output_nc = opt.output_nc
        z_nc = opt.z_nc
        zx_nc = z_nc + input_nc

        # resnet blocks
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(zx_nc, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # final conv: z_nc + x_nc -> x_nc
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(zx_nc, input_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, z, x):
        zz = torch.exp(z)
        #xx = F.dropout(x, p=self.x_drop, training=True)
        xx = x # NOTE: do dropout externally
        zx = torch.cat([zz, xx], 1)

        if self.gpu_ids and isinstance(zx.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.model, zx, self.gpu_ids)
        else:
            x = self.model(zx)
        return x


# net Discriminator
from math import ceil
class NLayerDiscriminator(nn.Module):
    def __init__(self, opt, gan_type='ls', norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        self.gan_type = gan_type
        self.gpu_ids = opt.gpu_ids
        input_nc = opt.input_nc
        ndf = opt.ndf
        n_layers = opt.n_layers_D
        kw = 4
        padw = int(ceil((kw-1)/2))

        # first conv-leakyReLU
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # n layers conv-BN-leakyReLU
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
        output = output.view(output.size(0), -1)
        return output
