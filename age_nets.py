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

################################
# data settings
################################
parser.add_argument('--dataset', type=str, default='cityscapesAB', help='chooses which dataset is loaded. [cityscapesAB | pascal | camvid]')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|no_resize]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
parser.add_argument('--ignore_index', type=int, default=-100, help='mask this class without contributing to nll_loss')
parser.add_argument('--unsup_portion', type=int, default=9, help='portion of unsupervised, range=0,...,10')
parser.add_argument('--portion_total', type=int, default=10, help='total portion of unsupervised, e.g., 10')
parser.add_argument('--unsup_sampler', type=str, default='sep', help='unif, sep, unif_ignore')

################################
# train settings
################################
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--checkpoints_dir', default='ckpt', help='folder to output images and model checkpoints')
parser.add_argument('--save_every', default=2, type=int, help='')
parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')
parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--gpu_ids', type=str, default='', help='gpu ids: e.g. 0; 0,2')
parser.add_argument('--update_HG', action='store_true', help='update net H & G')
parser.add_argument('--update_D', action='store_true', help='update net D')
parser.add_argument('--update_F', action='store_true', help='update net F')

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
opt.checkpoints_dir = opt.checkpoints_dir + '/' + opt.name
opt.update_H = True if opt.update_HG else False
opt.update_G = opt.update_H

assert(opt.unsup_sampler == 'sep')
assert(opt.unsup_portion > 0)

def get_opt():
    return opt

#########################################################################
def normalize_(x, dim=1):
    x.div_(x.norm(2, dim=dim, keepdim=True).expand_as(x))

def normalize(x, dim=1):
    return x.div(x.norm(2, dim=dim, keepdim=True).expand_as(x))

def var(x, dim=0):
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)

def populate_xy(x, y_int, dataloader, opt):
    # x, y are Variable
    AtoB = opt.which_direction == 'AtoB'
    real_cpu = dataloader.next()
    if x is not None:
        x_cpu = real_cpu['A' if AtoB else 'B']
        #x.data.resize_(x_cpu.size()).copy_(x_cpu)
        assert(x.data.size() == x_cpu.size())
        x.data.copy_(x_cpu)
    if y_int is not None:
        y_cpu = real_cpu['B' if AtoB else 'A']
        y_int.data.copy_(y_cpu)
    #if opt.DEBUG:
    #    print(real_cpu['A_paths'])

def populate_z(z, opt):
    z.data.resize_(opt.batchSize, opt.nz, 1, 1)
    z.data.normal_(0, 1)
    if opt.noise == 'sphere':
        normalize_(z.data)

def create_losses(opt):
    CE = torch.nn.NLLLoss2d(ignore_index=opt.ignore_index)
    L1 = torch.nn.L1Loss()

    if len(opt.gpu_ids) > 0:
        CE = CE.cuda(opt.gpu_ids[0])
        L1 = L1.cuda(opt.gpu_ids[0])

    def _loss_KL(mu, logvar, num_pixs):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= num_pixs
        return KLD
    return CE, L1, _loss_KL


#########################################################################
from models import networks
from models.networks import weights_init
from models.style_transform_resnet import StyleTransformResNet

# net X -> Y: F(x)
class GX2Y(nn.Module):
    def __init__(self, opt, temperature):
        super(GX2Y, self).__init__()
        self.gpu_ids = opt.gpu_ids
        self.temperature = temperature

        #self.softmax = nn.Softmax2d()
        self.logsoftmax = nn.LogSoftmax()
        self.resnet = StyleTransformResNet(opt.input_nc, opt.output_nc, opt.ngf,
                            norm_layer=nn.BatchNorm2d, use_dropout=opt.use_dropout, n_blocks=9,
                            gpu_ids=opt.gpu_ids, last_layer='softmax')

        def _forward(x):
            log_probs = self.resnet(x) # logsoftmax
            log_probs = self.reparameterize(log_probs)
            return log_probs

        self.model = _forward

    def reparameterize(self, log_probs, eps=1e-20):
        if self.training:
            # Gumbel(0,1): -log( -log(U + eps) + eps )
            noise = torch.rand(log_probs.size())
            noise.add_(eps).log_().neg_()
            noise.add_(eps).log_().neg_()
            if len(self.gpu_ids) > 0:
                noise = noise.cuda()
            noise = Variable(noise)
            sample_y = (log_probs + noise) / self.temperature
            return self.logsoftmax(sample_y)
        else:
            return log_probs

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class BranchX(nn.Module):
    def __init__(self, nc, ngf=64, gpu_ids=[]):
        super(BranchX, self).__init__()
        self.gpu_ids = gpu_ids

        # output size = (ngf*8) x 1 x 1
        self.model = nn.Sequential(
            # input size = nStates x 32 x 32
            nn.Conv2d(nc,      ngf,     4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # input size = (ngf) x 16 x 16
            nn.Conv2d(ngf,     ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # input size = (ngf*2) x 8 x 8
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # input size = (ngf*4) x 4 x 4
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AvgPool2d(2)
        )

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# net X -> Z: H(x)
class GX2Z(nn.Module):
    def __init__(self, nx, nz, ngf=64, gpu_ids=[]):
        super(GX2Z, self).__init__()
        self.gpu_ids = gpu_ids

        self.conv_x = BranchX(nx, ngf, gpu_ids)
        self.mu_op = nn.Linear(ngf * 8, nz)
        self.logvar_op = nn.Linear(ngf * 8, nz)

        def reparameterize():
            if self.training:
                std = self.logvar.mul(0.5).exp_()
                eps = Variable(std.data.new(std.size()).normal_())
                return eps.mul(std).add_(self.mu)
            else:
                return self.mu

        def _forward(x):
            h = self.conv_x(x) # B x (ngf*8) x 1 x 1
            h = h.view(-1, ngf * 8)
            self.mu, self.logvar = self.mu_op(h), self.logvar_op(h)
            h = reparameterize() # B x nz
            return h

        self.model = _forward

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# net Y, Z -> X: G(Y,Z)
class GYZ2X(nn.Module):
    def __init__(self, nx, ny, nz, ngf=64, gpu_ids=[]):
        super(GYZ2X, self).__init__()
        self.gpu_ids = gpu_ids

        # out size = (ngf*4) x 8 x 8
        self.branch_z = nn.Sequential(
            # input_size = nz x 1 x 1
            nn.ConvTranspose2d(nz,      ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # input size = (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # out size = (ngf*4) x 8 x 8
        self.branch_y = nn.Sequential(
            # input size = nStates x 32 x 32
            nn.Conv2d(ny,      ngf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # input size = (ngf) x 16 x 16
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # out size = nx x 32 x 32
        self.branch_combined = nn.Sequential(
            # input size = (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # input size = (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # input size = (ngf*2) x 32 x 32
            nn.Conv2d(ngf * 2, nx, 1, bias=True),
            nn.Tanh()
        )

        def _forward(input):
            y, z = input
            output_y = self.branch_y(y)
            output_z = self.branch_z(z)
            output = torch.cat((output_z, output_y), dim=1)
            output = self.branch_combined(output)
            return output

        self.model = _forward

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# net X, Y, Z -> score/feat: D(X,Y,Z)
class DXYZ(nn.Module):
    def __init__(self, nx, ny, nz, ngf, ndf, gpu_ids=[]):
        super(DXYZ, self).__init__()
        self.gpu_ids = gpu_ids

        self.conv_x = BranchX(nx, ndf, gpu_ids)
        self.conv_y = BranchX(ny, ndf, gpu_ids)

        self.branch_combined = nn.Sequential(
            nn.Linear(ngf*8+ngf*8+nz, ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf * 2, 1),
            #nn.Sigmoid()
        )

        def _forward(input):
            x, y, z = input
            output_x = self.conv_x(x).squeeze()
            output_y = self.conv_y(y).squeeze()
            output = torch.cat((output_x, output_y, z), dim=1)
            score = self.branch_combined(output)
            return score

        self.model = _forward

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input[0].data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        return output

