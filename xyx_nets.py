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
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--drop_lr', default=5, type=int, help='')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')

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
parser.add_argument('--stage', type=str, default='F', help='e.g. F:2,G:1,D:0')
parser.add_argument('--lambda_x', type=float, default=1.0, help='coeff of L1 and GAN')
parser.add_argument('--lambda_y', type=float, default=1.0, help='coeff of CE')
parser.add_argument('--lrFGD', type=str, default='1e-4,1e-4,1e-4', help='lrF,lrG,lrD')

################################
# model settings
################################
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--widthSize', type=int, default=256, help='crop to this width')
parser.add_argument('--heightSize', type=int, default=256, help='crop to this height')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=20, help='# of output image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--noise', default='sphere', help='normal|sphere')
parser.add_argument('--n_layers_D', type=int, default=3, help='')
parser.add_argument('--n_layers_F', type=int, default=9, help='')
parser.add_argument('--n_layers_G', type=int, default=9, help='')
parser.add_argument('--archD', type=str, default='patch', help='')
parser.add_argument('--archF', type=str, default='drn_d_22', help='')
parser.add_argument('--archG', type=str, default='style_transform', help='')

################################
# external
################################
parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--port', type=int, default=8097, help='port of visdom')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

# opt
opt = parser.parse_args()
opt.isTrain = True
opt.name += '_%s_b%d/stage%s/lrFGD%s_lbX%.3f' % (opt.dataset, opt.batchSize, opt.stage, opt.lrFGD, opt.lambda_x)

opt.updates = {k_v.split(':')[0]:int(k_v.split(':')[1]) for k_v in opt.stage.split(',')}
opt.lrFGD = {k:float(lr) for k,lr in zip(opt.updates.keys(), opt.lrFGD.split(','))}

assert(opt.unsup_sampler == 'sep')
#assert(opt.unsup_portion > 0)

def get_opt():
    return opt

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
    y = torch.FloatTensor(opt.batchSize, opt.output_nc, opt.heightSize, opt.widthSize)
    if len(opt.gpu_ids) > 0:
        y = y.cuda()
    y.zero_().scatter_(1, y_temp, 1)
    return y

def gumbel_softmax(log_probs, temperature=1.0, gpu_ids=[], eps=1e-20):
    ''' log_probs is a Variable '''
    # Gumbel(0,1): -log( -log(U + eps) + eps )
    noise = torch.rand(log_probs.size())
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    if len(gpu_ids) > 0:
        noise = noise.cuda()
    noise = Variable(noise)
    return nn.LogSoftmax()((log_probs + noise) / temperature)

def noise_log_y(y, temperature=1.0, gpu_ids=[], eps=1e-20):
    ''' y is a Variable '''
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


# net X -> Y: F(x)
class FX2Y(nn.Module):
    def __init__(self, opt, temperature):
        super(FX2Y, self).__init__()
        self.gpu_ids = opt.gpu_ids
        self.temperature = temperature

        #self.softmax = nn.Softmax2d()
        self.logsoftmax = nn.LogSoftmax()

        if opt.archF == 'style_transform':
            from models.style_transform_resnet import StyleTransformResNet
            self.resnet = StyleTransformResNet(opt.input_nc, opt.output_nc, opt.ngf,
                            norm_layer=nn.BatchNorm2d, use_dropout=opt.use_dropout, n_blocks=opt.n_layers_F,
                            gpu_ids=opt.gpu_ids, last_layer='softmax')
            self.resnet.apply(weights_init)
        elif opt.archF == 'resnet50_fcn':
            from models.resnet50_fcn import ResNet50FCN
            self.resnet = ResNet50FCN(opt.output_nc, freeze_batch_norm=False, gpu_ids=opt.gpu_ids)
        elif 'drn' in opt.archF:
            from models.drn import DRNSeg
            self.resnet = DRNSeg(opt.output_nc, opt.archF, gpu_ids=opt.gpu_ids, pretrained=True, use_torch_up=False)
        else:
            raise ValueError('%s not recognized!' % opt.archF)

        def _forward(x):
            log_probs = self.resnet(x) # logsoftmax
            noisy_log_probs = self.reparameterize(log_probs)
            return noisy_log_probs

        self.model = _forward

    def reparameterize(self, log_probs):
        if self.training:
            return gumbel_softmax(log_probs, self.temperature, self.gpu_ids)
        else:
            return log_probs

    def forward(self, input):
        y_hat = self.model(input)
        return y_hat

# net Y -> X: G(y)
class GY2X(nn.Module):
    def __init__(self, opt):
        super(GY2X, self).__init__()
        self.gpu_ids = opt.gpu_ids

        if opt.archG == 'style_transform':
            from models.style_transform_resnet import StyleTransformResNet
            self.model = StyleTransformResNet(opt.output_nc, opt.input_nc, opt.ngf,
                                              norm_layer=nn.BatchNorm2d, use_dropout=opt.use_dropout, n_blocks=opt.n_layers_G,
                                              gpu_ids=opt.gpu_ids, last_layer='tanh')
        elif opt.archG == 'unet_128': # NOTE: unet works only for 128x128 or 256x256
            from models.u_net import UnetGenerator
            self.model = UnetGenerator(opt.output_nc, opt.input_nc, 7, opt.ngf, norm_layer=nn.BatchNorm2d,
                                       use_dropout=opt.use_dropout, gpu_ids=opt.gpu_ids)
        elif opt.archG == 'unet_256':
            from models.u_net import UnetGenerator
            self.model = UnetGenerator(opt.output_nc, opt.input_nc, 8, opt.ngf, norm_layer=nn.BatchNorm2d,
                                       use_dropout=opt.use_dropout, gpu_ids=opt.gpu_ids)
        else:
            raise ValueError('%s not recognized!' % opt.archG)

    def forward(self, input):
        x_hat = self.model(input)
        return x_hat

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
