import os
import time
import subprocess
import sys
import shutil
import re
import numpy as np
from util import util
import argparse

pybin = sys.executable

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
parser.add_argument('--stage', type=str, default='F', help='F, GD, F2, FGD')
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
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')

################################
# external
################################
parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--port', type=int, default=8097, help='port of visdom')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

# opt
opt = parser.parse_args()

#----------------------------------------------------------------
opt.parallel = True
if opt.parallel:
    call_func = subprocess.Popen
else:
    call_func = subprocess.call

def batch_train(lrs, lambda_xs, stage_str):
    re_miou = re.compile("\'Mean IoU\': \d+\.\d+")
    max_ious = np.zeros((len(lrs), len(lambda_xs)))

    for i,lr in enumerate(lrs):

        if isinstance(lr, str):
            lrFGD = lr
        else:
            lrFGD = '%.1e,%.1e,%.1e' % (lr, lr, lr)

        for j,lb in enumerate(lambda_xs):
            # TODO: try dropout
            cmd = "%s xyx_train.py --name %s --checkpoints_dir ./checkpoints --output_nc %d --dataset %s --batchSize %d " \
                  "--heightSize %d --widthSize %d --niter %d --drop_lr %d --resize_or_crop %s " \
                  "--ignore_index %d --unsup_portion %d --portion_total %d --unsup_sampler %s " \
                  "--port %d --gpu_ids %s --lrFGD %s --lambda_x %.3f --stage %s" \
                % (pybin, opt.name, opt.output_nc, opt.dataset, opt.batchSize, \
                   opt.heightSize, opt.widthSize, opt.niter, opt.drop_lr, opt.resize_or_crop, \
                   opt.ignore_index, opt.unsup_portion, opt.portion_total, opt.unsup_sampler, \
                   opt.port, opt.gpu_ids, lrFGD, lb, stage_str)
            print(cmd + '\n')

            outfile = "./logs/%s_%s_b%d/stage%s_lrFGD%s_lb%.3f.log" % (opt.name, opt.dataset, opt.batchSize, stage_str, lrFGD, lb)
            util.mkdir( os.path.dirname(outfile) ) # mkdir if not exist

            with open(outfile, 'w') as output_f:
                p = call_func(cmd, stdin=open('/dev/null'), stdout=output_f, stderr=output_f, shell=True)
                #p = call_func(cmd, stdin=open('/dev/null'), stdout=subprocess.PIPE, stderr=output_f, shell=True)
                p.wait()

            # TODO: 3. check net load for --update_D
            lr_lb_name = opt.name + '_%s_b%d/stage%s/lrFGD%s_lbX%.3f' % (opt.dataset, opt.batchSize, stage_str, lrFGD, lb)
            logfile = os.path.join(opt.checkpoints_dir, lr_lb_name, 'loss_log.txt')
            assert(os.path.exists(logfile))
            with open(logfile, 'r') as log_f:
                ious = re.findall(re_miou, log_f.read())
                ious = [float(iou.split(': ')[1]) for iou in ious]
                max_ious[i,j] = max(ious)
    return max_ious

#----------------------------------------------------------------
# stage F:2,G:0,D:0 --> lr_F
lrs = [1e-2, 1e-3, 1e-4]
stage_str = 'F:2,G:0,D:0'

F_max_ious = batch_train(lrs, [1.0], stage_str)

#arg_lr_F = np.unravel_index(F_max_ious.argmax(), F_max_ious.shape)
arg_lr_F = F_max_ious.argmax(axis=0)
lr_F = lrs[arg_lr_F[0]] # **opt lr_F
#lr_F = 1e-3 # DEBUG

lrFGD = '%.1e,%.1e,%.1e' % (lr_F, lr_F, lr_F)
stageF_name = opt.name + '_%s_b%d/stage%s/lrFGD%s_lbX%.3f' % (opt.dataset, opt.batchSize, stage_str, lrFGD, 1.0)
F_stageF_path = os.path.join(opt.checkpoints_dir, stageF_name, 'netF.pth')

print('\n==> Stage F: mIoU = %f by lr_F = %.1e\n' % (F_max_ious.max(), lr_F))

#ipdb> F_max_ious
#array([[ 0.73601473],
#       [ 0.73649235],
#       [ 0.68873321]])

# 'Mean IoU': 0.72510022394608131

#----------------------------------------------------------------
# stage F:1,G:2,D:2: freeze F, update G & D --> lr_GD
# TODO: check L1 rather than IoU
#lrs = [1e-3, 1e-4, 1e-5]
lrs = [1e-4] # DEBUG
stage_str = 'F:1,G:2,D:2'

for lr in lrs:
    lrFGD = '%.1e,%.1e,%.1e' % (lr, lr, lr)
    stageGD_name = opt.name + '_%s_b%d/stage%s/lrFGD%s_lbX%.3f' % (opt.dataset, opt.batchSize, stage_str, lrFGD, 1.0)
    F_stageGD_path = os.path.join(opt.checkpoints_dir, stageGD_name, 'netF.pth')
    util.mkdir( os.path.dirname(F_stageGD_path) ) # mkdir if not exist
    shutil.copy(F_stageF_path, F_stageGD_path)

GD_max_ious = batch_train(lrs, [1.0], stage_str)
arg_lr_GD = GD_max_ious.argmax(axis=0)
lr_GD = lrs[arg_lr_GD[0]] # **opt lr_GD
#lr_GD = 1e-4 # DEBUG

lrFGD = '%.1e,%.1e,%.1e' % (lr_GD, lr_GD, lr_GD)
stageGD_name = opt.name + '_%s_b%d/stage%s/lrFGD%s_lbX%.3f' % (opt.dataset, opt.batchSize, stage_str, lrFGD, 1.0)
G_stageGD_path = os.path.join(opt.checkpoints_dir, stageGD_name, 'netG.pth')
D_stageGD_path = os.path.join(opt.checkpoints_dir, stageGD_name, 'netD.pth')

print('\n==> Stage GD: mIoU = %f by lr_GD = %.1e\n' % (GD_max_ious.max(), lr_GD))

#----------------------------------------------------------------
# stage F:2,G:1,D:1: freeze G & D, update F with lr_F --> lambda_x
lambda_xs = [100, 10, 1, 1e-1, 1e-2]
lrFGD = '%.1e,%.1e,%.1e' % (lr_F, lr_GD, lr_GD)
stage_str = 'F:2,G:1,D:1'

for lb in lambda_xs:
    stageF2_name = opt.name + '_%s_b%d/stage%s/lrFGD%s_lbX%.3f' % (opt.dataset, opt.batchSize, stage_str, lrFGD, lb)
    F_stageF2_path = os.path.join(opt.checkpoints_dir, stageF2_name, 'netF.pth')
    G_stageF2_path = os.path.join(opt.checkpoints_dir, stageF2_name, 'netG.pth')
    D_stageF2_path = os.path.join(opt.checkpoints_dir, stageF2_name, 'netD.pth')
    util.mkdir( os.path.dirname(F_stageF2_path) ) # mkdir if not exist
    util.mkdir( os.path.dirname(G_stageF2_path) ) # mkdir if not exist
    util.mkdir( os.path.dirname(D_stageF2_path) ) # mkdir if not exist
    shutil.copy(G_stageGD_path, G_stageF2_path)
    shutil.copy(D_stageGD_path, D_stageF2_path)
    shutil.copy(F_stageF_path,  F_stageF2_path)

F2_max_ious = batch_train([lrFGD], lambda_xs, stage_str)
arg_lb_x = F2_max_ious.argmax(axis=1)
lb_x = lambda_xs[arg_lb_x[0]] # **opt lb_x
# lb_x = 10 # DEBUG

stageF2_name = opt.name + '_%s_b%d/stage%s/lrFGD%s_lbX%.3f' % (opt.dataset, opt.batchSize, stage_str, lrFGD, lb_x)
F_stageF2_path = os.path.join(opt.checkpoints_dir, stageF2_name, 'netF.pth')

print('\n==> Stage F2: mIoU = %f by lb_x = %.3f\n' % (F2_max_ious.max(), lb_x))

#----------------------------------------------------------------
# stage F:2,G:2,D:2: update all nets with lr_F --> lambda_x
stage_str = 'F:2,G:2,D:2'
lrFGD = '%.1e,%.1e,%.1e' % (lr_F, lr_GD / lb_x, lr_GD / lb_x)
stageFGD_name = opt.name + '_%s_b%d/stage%s/lrFGD%s_lbX%.3f' % (opt.dataset, opt.batchSize, stage_str, lrFGD, lb_x)

F_stageFGD_path = os.path.join(opt.checkpoints_dir, stageFGD_name, 'netF.pth')
G_stageFGD_path = os.path.join(opt.checkpoints_dir, stageFGD_name, 'netG.pth')
D_stageFGD_path = os.path.join(opt.checkpoints_dir, stageFGD_name, 'netD.pth')
util.mkdir( os.path.dirname(F_stageFGD_path) ) # mkdir if not exist
util.mkdir( os.path.dirname(G_stageFGD_path) ) # mkdir if not exist
util.mkdir( os.path.dirname(D_stageFGD_path) ) # mkdir if not exist
shutil.copy(G_stageGD_path, G_stageFGD_path)
shutil.copy(D_stageGD_path, D_stageFGD_path)
shutil.copy(F_stageF2_path, F_stageFGD_path)

FGD_max_ious = batch_train([lrFGD], [lb_x], stage_str)

print('\n==> Stage FGD: mIoU = %f\n' % (FGD_max_ious.max()))

# ==> Stage FGD: mIoU = 0.737043
