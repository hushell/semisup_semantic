from __future__ import print_function
import time
import os
import torch
from itertools import izip

import argparse
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

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
# train settings
################################
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--dataset', type=str, default='cityscapesAB', help='chooses which dataset is loaded. [cityscapesAB | pascal | camvid]')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|no_resize]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
parser.add_argument('--ignore_index', type=int, default=-100, help='mask this class without contributing to nll_loss')

parser.add_argument('--unsup_portion', type=int, default=9, help='portion of unsupervised, range=0,...,10')
parser.add_argument('--portion_total', type=int, default=10, help='total portion of unsupervised, e.g., 10')
parser.add_argument('--unsup_sampler', type=str, default='sep', help='unif, sep, unif_ignore')
parser.add_argument('--checkpoints_dir', default='ckpt', help='folder to output images and model checkpoints')
parser.add_argument('--save_every', default=5, type=int, help='')
parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')
parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--gpu_ids', type=str, default='', help='gpu ids: e.g. 0; 0,2')

################################
# model settings
################################
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--widthSize', type=int, default=256, help='crop to this width')
parser.add_argument('--heightSize', type=int, default=256, help='crop to this height')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=20, help='# of output image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--net_chp', default='', help="path to nets (to continue training)")
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

assert(opt.unsup_sampler == 'sep')
assert(opt.unsup_portion > 0)

# gpu id
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids # absolute ids
if len(opt.gpu_ids) > 0:
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
opt.gpu_ids = range(0,len(opt.gpu_ids)) # new range starting from 0

#########################################################################
from data.data_loader import CreateDataLoader,InfiniteDataLoader,XYDataLoader
from util.visualizer import Visualizer

# data_loaders
opt.phase = 'val'
opt.isTrain = False
val_loader = CreateDataLoader(opt)
opt = val_loader.update_opt(opt)

opt.phase = 'train'
opt.isTrain = True
paired_loader = XYDataLoader(opt, is_paired=True)
x_loader = XYDataLoader(opt, is_paired=False) # will use x only
y_loader = XYDataLoader(opt, is_paired=False) # will use y only

# wrap with infinite loader
val_loader = InfiniteDataLoader(val_loader)
paired_loader = InfiniteDataLoader(paired_loader)
x_loader = InfiniteDataLoader(x_loader)
y_loader = InfiniteDataLoader(y_loader)

# Visualizer
visualizer = Visualizer(opt)

#########################################################################
from age_nets import *
TAU0 = 1.0

# networks
net =dict()

net['F'] = GX2Y(opt, temperature=TAU0)
net['H'] = GX2Z(opt.input_nc, opt.nz, opt.ngf, opt.gpu_ids)
net['G'] = GYZ2X(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, opt.gpu_ids)
net['D'] = DXYZ(opt.input_nc, opt.output_nc, opt.nz, opt.ndf, opt.gpu_ids)

for k in net.keys():
    net[k].apply(weights_init)
    net[k].train()
    if opt.net_chp != '':
        net[k].load_state_dict(torch.load(opt.net_chp + '_' + k).state_dict())

#########################################################################
# variables
x = torch.FloatTensor(opt.batchSize, opt.input_nc, opt.heightSize, opt.widthSize)
y = torch.FloatTensor(opt.batchSize, opt.output_nc, opt.heightSize, opt.widthSize)
y_int = torch.LongTensor(opt.batchSize, opt.heightSize, opt.widthSize)
z = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)

if len(opt.gpu_ids) > 0:
    x = x.cuda(opt.gpu_ids[0])
    y = y.cuda(opt.gpu_ids[0])
    y_int = y_int.cuda(opt.gpu_ids[0])
    z = z.cuda(opt.gpu_ids[0])
    for k in net.keys():
        net[k].cuda(opt.gpu_ids[0])

x = Variable(x)
y = Variable(y)
y_int = Variable(y_int)
z = Variable(z)

#########################################################################
# optimizers
optimizer = dict()
for k in net.keys():
    optimizer[k] = optim.Adam(net[k].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def adjust_lr(epoch):
    if epoch % opt.drop_lr == (opt.drop_lr - 1):
        opt.lr /= 2
        for k in optimizer.keys():
            for param_group in optimizer[k].param_groups:
                param_group['lr'] = opt.lr

# losses
CE, L1, KL =  create_losses(opt)

#########################################################################
D_ITERS = 0 # TODO: dynamic trick
G_ITERS = 1
CLAMP_LOW = -0.01
CLAMP_UPP = 0.01
ANNEAL_RATE=0.00003
MIN_TEMP=0.5
LAMBDA_CE = 10.0
num_pixs = opt.batchSize * opt.heightSize * opt.widthSize

def noise_y(y):
    log_probs_gt = torch.log(y + 1e-9)
    y = net['F'].reparameterize(log_probs_gt) # add noise
    return y

def populate_xyz_hat():
    # X -> Y, Z
    populate_xy(x, None, x_loader, opt)
    z_hat = net['H'](x)
    y_hat = net['F'](x)

    # Y, Z -> X
    populate_xy(None, y_int, y_loader, opt)
    one_hot(y, y_int)
    y = noise_y(y)
    populate_z(z, opt)
    x_hat = net['G']( (y,z) )

    return x_hat,y_hat,z_hat

from util.meter import SegmentationMeter
def evaluation():
    x = torch.FloatTensor(1, opt.input_nc, opt.heightSize, opt.widthSize)
    y_int = torch.LongTensor(1, opt.heightSize, opt.widthSize)
    if len(opt.gpu_ids) > 0:
        x = x.cuda(opt.gpu_ids[0])
        y_int = y_int.cuda(opt.gpu_ids[0])
    x = Variable(x)
    y_int = Variable(y_int)

    net['F'].eval()
    eval_stats = SegmentationMeter(n_class=opt.output_nc, ignore_index=opt.ignore_index)
    E_loss_CE = []
    start_time = time.time()
    for i in range(len(val_loader)):
        populate_xy(x, y_int, val_loader, opt)
        y_hat = net['F'](x)
        E_loss_CE.append( CE(y_hat, y_int) )
        logits = y_hat.data.cpu().numpy()
        pred = logits.argmax(1) # NCHW -> NHW
        gt = y_int.data.cpu().numpy()
        eval_stats.update_confmat(gt, pred)

    print('EVAL ==> average CE = %.3f' % (sum(E_loss_CE).data[0] / len(val_loader)))
    eval_results = eval_stats.get_eval_results()
    msg = 'EVAL [%d images] ==> \t Time Taken: %.2f sec: %s\n' % \
                (len(val_loader), time.time()-start_time, eval_results[0])
    msg += 'Per-class IoU:\n'
    msg += ''.join(['%s: %.2f\n' % (cname,ciu)
                    for cname,ciu in zip(val_loader.dataloader.dataset.label2name, eval_results[1])])
    print(msg)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s' % msg)

    net['F'].train()

# main loop
stats = {}
for epoch in range(opt.start_epoch, opt.niter):

    # on begin epoch
    adjust_lr(epoch)
    net['F'].temperature = np.maximum(TAU0*np.exp(-ANNEAL_RATE*epoch),MIN_TEMP)
    epoch_start_time = time.time()

    for i in range(len(x_loader)):
        iter_start_time = time.time()

        # ---------------------------
        #        Optimize over D
        # ---------------------------
        for d_i in range(D_ITERS):
            x_hat,y_hat,z_hat = populate_xyz_hat()

            E_q_D = net['D']( [x,y_hat.detach(),z_hat.detach()] ).mean()
            E_p_D = net['D']( [x_hat.detach(),y.detach(),z.view_as(z_hat)] ).mean()
            d_loss = -1.0 * (E_q_D - E_p_D).pow(2)

            optimizer['D'].zero_grad()
            d_loss.backward()
            optimizer['D'].step()

            # clamp parameters to a cube
            for p in net['D'].parameters():
                p.data.clamp_(CLAMP_LOW, CLAMP_UPP)

            #stats['E_q_D'] = E_q_D.data[0]
            #stats['E_p_D'] = E_p_D.data[0]
            stats['D'] = -d_loss.data[0]

        # ---------------------------
        #        Optimize over F,H,G
        # ---------------------------
        for g_i in range(G_ITERS):
            g_losses = []

            def update_FGH():
                optimizer['F'].zero_grad()
                optimizer['G'].zero_grad()
                optimizer['H'].zero_grad()
                sum(g_losses).backward()
                optimizer['F'].step()
                optimizer['G'].step()
                optimizer['H'].step()

            # paired X, Y
            populate_xy(x, y_int, paired_loader, opt)
            z_hat = net['H'](x)
            y_hat = net['F'](x)

            paired_loss_CE = CE(y_hat, y_int)
            g_losses.append( LAMBDA_CE * paired_loss_CE ) # CE

            paired_loss_KL = KL(net['H'].mu, net['H'].logvar, num_pixs)
            g_losses.append( paired_loss_KL ) # KL

            one_hot(y, y_int)
            y = noise_y(y)
            x_hat = net['G']( [y,z_hat.view_as(z)] )
            paired_loss_L1 = L1(x_hat, x)
            g_losses.append( paired_loss_L1 ) # L1

            update_FGH()

            stats['P_CE'] = paired_loss_CE.data[0]
            stats['P_KL'] = paired_loss_KL.data[0]
            stats['P_L1'] = paired_loss_L1.data[0]

            ## X, Y augmented
            #x_hat,y_hat,z_hat = populate_xyz_hat()

            #x_tilde = net['G']([(y_hat,z_hat.view_as(z)] ) # x -> y_hat, z_hat -> x_tilde
            #aug_loss_L1 = L1(x_tilde, x)
            #g_losses.append( aug_loss_L1 ) # L1

            #y_tilde = net['F'](x_hat) # y,z -> x_hat -> y_tilde
            #aug_loss_CE = CE(y_tilde, y)
            #g_losses.append( aug_loss_CE ) # CE

            #z_tilde = net['H'](x_hat) # y,z -> x_hat -> z_tilde
            #aug_loss_KL = (z - net['H'].mu).pow(2).div(net['H'].logvar.exp().mul(2)).mean()
            #g_losses.append( aug_loss_KL ) # Gaussian

            #E_q_G = net['D']( [x,y_hat,z_hat] ).mean()
            #E_p_G = net['D']( [x_hat,y,z.view_as(z_hat)] ).mean()
            #g_loss = (E_q_G - E_p_G).pow(2)
            #g_losses.append( g_loss )

            #update_FGH()

            #stats['A_CE'] = aug_loss_CE.data[0]
            ##stats['A_KL'] = aug_loss_KL.data[0]
            #stats['A_L1'] = aug_loss_L1.data[0]
            ##stats['E_q_G'] = E_q_G.data[0]
            ##stats['E_p_G'] = E_p_G.data[0]
            #stats['G'] = g_loss.data[0]
            stats['A_CE'] = 0
            stats['A_L1'] = 0
            stats['G'] = 0
            stats['D'] = 0

        # time spent per sample
        t = (time.time() - iter_start_time) / opt.batchSize

        if i % 10 == 0:
            print('[{epoch}/{nepoch}][{iter}/{niter}] '
                  'D/G: {D:.3f}/{G:.3f} '
                  'P_CE/A_CE: {P_CE:.3f}/{A_CE:.3f} '
                  'P_L1/A_L1: {P_L1:.3f}/{A_L1:.3f} '
                  ''.format(epoch=epoch,
                            nepoch=opt.niter,
                            iter=i,
                            niter=len(x_loader),
                            **stats))

    # on end epoch
    print('===> End of epoch %d / %d \t Time Taken: %.2f sec\n' % \
                (epoch, opt.niter, time.time() - epoch_start_time))

    # evaluation & save
    if epoch % opt.save_every == 0:
        evaluation()

