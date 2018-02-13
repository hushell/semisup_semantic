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
                  "--heightSize %d --widthSize %d --start_epoch %d --niter %d --drop_lr %d --resize_or_crop %s " \
                  "--ignore_index %d --unsup_portion %d --portion_total %d --unsup_sampler %s " \
                  "--port %d --gpu_ids %s --lrFGD %s --lambda_x %.3f --stage %s" \
                % (pybin, opt.name, opt.output_nc, opt.dataset, opt.batchSize, \
                   opt.heightSize, opt.widthSize, opt.start_epoch, opt.niter, opt.drop_lr, opt.resize_or_crop, \
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

import re

re_eval_miou = re.compile("\'Mean IoU\': \d+\.\d+")
re_eval_CE   = re.compile("average CE = \d+\.\d+")
re_DG   = re.compile("D/G: \d+\.\d+/\d+\.\d+")
re_CE   = re.compile("P_CE/A_CE: \d+\.\d+/\d+\.\d+")
re_L1   = re.compile("P_L1/A_L1: \d+\.\d+/\d+\.\d+")

valid_experiments = range(2,25)
#invalid_experiments = [10, 14, 18, 22]
invalid_experiments = [24]
valid_experiments = [x for x in valid_experiments if x not in invalid_experiments]
print(valid_experiments)


def load_log(name, stage_str, lrFGD, lb):
    outfile = "../logs/%s_%s_b%d/stage%s_lrFGD%s_lb%.3f.log" \
              % (name, opt.dataset, opt.batchSize, stage_str, lrFGD, lb)
    #print(outfile)
    #assert( os.path.exists(outfile) )
    if not os.path.exists(outfile):
        return [-9999]
    with open(outfile, 'r') as log_f:
        ious = re.findall(re_eval_miou, log_f.read())
        ious = [float(iou.split(': ')[1]) for iou in ious]
        return ious


def load_max_over_lb(name, lrFGD, lambda_xs, stage_str):
    all_ious = []
    amax = -1
    max_iou = -1

    for j,lb in enumerate(lambda_xs):
        ious = load_log(name, stage_str, lrFGD, lb)
        all_ious.append(ious)
        if max(ious) > max_iou:
            max_iou = max(ious)
            amax = j
    #print('%s: max_iou = %f, amax = %d' % (name, max_iou, amax))
    return all_ious[amax], amax


def load_max_over_lr(name, lst_lrFGD, lb, stage_str):
    all_ious = []
    amax = -1
    max_iou = -1

    for j,lrFGD in enumerate(lst_lrFGD):
        ious = load_log(name, stage_str, lrFGD, lb)
        all_ious.append(ious)
        if max(ious) > max_iou:
            max_iou = max(ious)
            amax = j
    #print('%s: max_iou = %f, amax = %d' % (name, max_iou, amax))
    return all_ious[amax], amax


#----------------------------------------------------------------
# stage F
stage_str = 'F:2,G:0,D:0'
lrs = [1e-2, 1e-3, 1e-4]
lb = 1.0

all_ious = []
all_lr_F = []

for j,i in enumerate(valid_experiments):
    name = '%s_%d' % (opt.name, i)
    lst_lrFGD = ['%.1e,%.1e,%.1e' % (lr,lr,lr) for lr in lrs]
    ious,amax = load_max_over_lr(name, lst_lrFGD, lb, stage_str)
    all_ious.append(ious)
    all_lr_F.append(lrs[amax])

F_ious = np.stack(all_ious, axis=0)


#----------------------------------------------------------------
# stage GD
stage_str = 'F:1,G:2,D:2'
lr_GD = 1e-4
all_lr_GD = [lr_GD] * len(valid_experiments)


#----------------------------------------------------------------
# stage F2
stage_str = 'F:2,G:1,D:1'
lambda_xs = [100, 10, 1, 1e-1, 1e-2]

all_ious = []
all_lb = []

for j,i in enumerate(valid_experiments):
    name = '%s_%d' % (opt.name, i)
    lrFGD = '%.1e,%.1e,%.1e' % (all_lr_F[j],all_lr_GD[j],all_lr_GD[j])
    ious,amax = load_max_over_lb(name, lrFGD, lambda_xs, stage_str)
    all_ious.append(ious)
    all_lb.append(lambda_xs[amax])

F2_ious = np.stack(all_ious, axis=0)


#----------------------------------------------------------------
# stage FGD
stage_str = 'F:2,G:2,D:2'
all_ious = []
gname = opt.name
for j,i in enumerate(valid_experiments):
    if i < 20:
        continue
    opt.name = '%s_%d' % (gname, i)
    lb = all_lb[j]
    lr_F = all_lr_F[j]
    lr_GD = all_lr_GD[j]
    lrFGD = '%.1e,%.1e,%.1e' % (lr_F, lr_GD / lb, lr_GD / lb)
    #ious = load_log(name, stage_str, lrFGD, lb)
    #all_ious.append(ious)

    FGD_max_ious = batch_train([lrFGD], [lb], stage_str)
    print('==> (%d) Stage FGD: mIoU = %f' % (j, FGD_max_ious.max()))

#FGD_ious = np.stack(all_ious, axis=0)
