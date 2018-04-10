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
parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
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
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--noise', default='sphere', help='normal|sphere')
parser.add_argument('--n_layers_D', type=int, default=3, help='')

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
            lrFGD = '%.1e,%.1e,%.1e,%.1e' % (lr, lr, lr, lr)

        for j,lb in enumerate(lambda_xs):
            cmd = "CUDA_VISIBLE_DEVICES=%s %s xz2yx_train.py --name %s --checkpoints_dir ./checkpoints --output_nc %d --dataset %s --batchSize %d " \
                  "--heightSize %d --widthSize %d --n_layers_D %d --start_epoch %d --niter %d --niter_decay %d --resize_or_crop %s " \
                  "--ignore_index %d --unsup_portion %d --portion_total %d --unsup_sampler %s --display_id %d " \
                  "--port %d --gpu_ids %d --lrs %s --lambdas %s --stage %s" \
                % (opt.gpu_ids, pybin, opt.name, opt.output_nc, opt.dataset, opt.batchSize, \
                   opt.heightSize, opt.widthSize, opt.n_layers_D, opt.start_epoch, opt.niter, opt.niter_decay, opt.resize_or_crop, \
                   opt.ignore_index, opt.unsup_portion, opt.portion_total, opt.unsup_sampler, opt.display_id, \
                   opt.port, 0, lrFGD, lb, stage_str) # NOTE: only lrFGD, lb, stage change
            print(cmd + '\n')

            outfile = './logs/%s_%s_b%d/stage%s_lr%s_lb%s.log' % (opt.name, opt.dataset, opt.batchSize, stage_str, lrFGD, lb)
            util.mkdir( os.path.dirname(outfile) ) # mkdir if not exist

            with open(outfile, 'w') as output_f:
                p = call_func(cmd, stdin=open('/dev/null'), stdout=output_f, stderr=output_f, shell=True)
                #p = call_func(cmd, stdin=open('/dev/null'), stdout=subprocess.PIPE, stderr=output_f, shell=True)
                p.wait()

            lr_lb_name = opt.name + '_%s_b%d/stage%s/lr%s_lb%s' % (opt.dataset, opt.batchSize, stage_str, lrFGD, lb)
            logfile = os.path.join(opt.checkpoints_dir, lr_lb_name, 'loss_log.txt')
            assert(os.path.exists(logfile))
            with open(logfile, 'r') as log_f:
                ious = re.findall(re_miou, log_f.read())
                ious = [float(iou.split(': ')[1]) for iou in ious]
                max_ious[i,j] = max(ious)
    return max_ious


#----------------------------------------------------------------
# stage X2Z:2,Z2Y:2,ZY2X:0,D:0
lrs = ['1e-3,1e-3,1e-4,1e-4']
lambdas = ['1e-0,1e-0']
stage_str = 'X2Z:2,Z2Y:2,ZY2X:0,D:0'

F_max_ious = batch_train(lrs, lambdas, stage_str)

arg_lr_F = F_max_ious.argmax(axis=0)
lr_F = lrs[arg_lr_F[0]] # **opt lr_F

stage_name = opt.name + '_%s_b%d/stage%s/lr%s_lb%s' % (opt.dataset, opt.batchSize, stage_str, lr_F, lambdas[0])
X2Z_path = os.path.join(opt.checkpoints_dir, stage_name, 'netX2Z.pth')
Z2Y_path = os.path.join(opt.checkpoints_dir, stage_name, 'netZ2Y.pth')

print('\n==> Stage paired: mIoU = %f by lr_F = %s\n' % (F_max_ious.max(), lr_F))


#----------------------------------------------------------------
# stage X2Z:2,Z2Y:2,ZY2X:2,D:2
stage_str = 'X2Z:2,Z2Y:2,ZY2X:2,D:2'

lrs = ['1e-4,1e-4,1e-4,1e-4']
lambdas = ['1e-1,1e-0']
stage_str = 'X2Z:2,Z2Y:2,ZY2X:2,D:2'
final_stage_name = opt.name + '_%s_b%d/stage%s/lr%s_lb%s' % (opt.dataset, opt.batchSize, stage_str, lrs[0], lambdas[0])

final_X2Z_path = os.path.join(opt.checkpoints_dir, final_stage_name, 'netX2Z.pth')
final_Z2Y_path = os.path.join(opt.checkpoints_dir, final_stage_name, 'netZ2Y.pth')
util.mkdir( os.path.dirname(final_X2Z_path) ) # mkdir if not exist
util.mkdir( os.path.dirname(final_Z2Y_path) ) # mkdir if not exist
shutil.copy(X2Z_path, final_X2Z_path)
shutil.copy(Z2Y_path, final_Z2Y_path)

FGD_max_ious = batch_train(lrs, lambdas, stage_str)

print('\n==> Stage FGD: mIoU = %f\n' % (FGD_max_ious.max()))

