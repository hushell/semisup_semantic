from __future__ import print_function
import time
import os
import torch
from itertools import izip
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from age_nets import *

opt = get_opt()
stage_str = ('F' if opt.update_F else '') + ('HG' if opt.update_HG else '') + ('D' if opt.update_D else '')

# gpu id
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids # absolute ids
if len(opt.gpu_ids) > 0:
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
opt.gpu_ids = range(0,len(opt.gpu_ids)) # new range starting from 0

#########################################################################
# data_loaders
from data.data_loader import CreateDataLoader,InfiniteDataLoader,XYDataLoader
from util.visualizer import Visualizer

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
paired_loader = InfiniteDataLoader(paired_loader)
x_loader = InfiniteDataLoader(x_loader)
y_loader = InfiniteDataLoader(y_loader)

# Visualizer
visualizer = Visualizer(opt)

#########################################################################
# networks
TAU0 = 1.0

net =dict()
net['F'] = GX2Y(opt, temperature=TAU0)
net['H'] = GX2Z(opt.input_nc, opt.nz, opt.ngf, opt.gpu_ids)
net['G'] = GYZ2X(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, opt.gpu_ids)
net['D'] = DXYZ(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, opt.ndf, opt.gpu_ids)

for k in net.keys():
    if len(opt.gpu_ids) > 0:
        net[k].cuda(opt.gpu_ids[0])

    net[k].apply(weights_init)

    # load if found saved weights
    weights_fpath = os.path.join(opt.checkpoints_dir, 'net%s_stage%s.pth' % (k,stage_str))
    if os.path.exists(weights_fpath):
        net[k].load_state_dict(torch.load(weights_fpath, map_location=lambda storage, loc: storage))

    # freeze some nets
    if getattr(opt, 'update_%s' % k):
        net[k].train()
        for param in net[k].parameters():
            param.requires_grad = True
    else:
        net[k].eval() # NOTE: to disable batchnorm updates
        for param in net[k].parameters():
            param.requires_grad = False

#########################################################################
# variables
x = torch.FloatTensor(opt.batchSize, opt.input_nc, opt.heightSize, opt.widthSize)
#y = torch.FloatTensor(opt.batchSize, opt.output_nc, opt.heightSize, opt.widthSize)
y_int = torch.LongTensor(opt.batchSize, opt.heightSize, opt.widthSize)
z = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)

if len(opt.gpu_ids) > 0:
    x = x.cuda(opt.gpu_ids[0])
    y_int = y_int.cuda(opt.gpu_ids[0])
    z = z.cuda(opt.gpu_ids[0])

#########################################################################
# optimizers
optimizer = dict()
for k in net.keys():
    #optimizer[k] = optim.Adam(filter(lambda p: p.requires_grad, net[k].parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
    if getattr(opt, 'update_%s' % k):
        optimizer[k] = optim.Adam(net[k].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def adjust_lr(epoch):
    #if epoch % opt.drop_lr == (opt.drop_lr - 1):
    #    opt.lr /= 2
    #    for k in optimizer.keys():
    #        for param_group in optimizer[k].param_groups:
    #            param_group['lr'] = opt.lr
    #    print('===> Start of epoch %d / %d \t lr = %.6f' % (epoch, opt.niter, opt.lr))
    print('===> Start of epoch %d / %d \t lr = %.6f' % (epoch, opt.niter, opt.lr))

# losses
CE, L1, KL = create_losses(opt)

#########################################################################
CLAMP_LOW = -0.01
CLAMP_UPP = 0.01
ANNEAL_RATE = 0.00003
MIN_TEMP = 0.5
LAMBDA_CE = 10.0
num_pixs = opt.batchSize * opt.heightSize * opt.widthSize

def populate_xyz_hat(temperature):
    # X
    populate_xy(x, None, x_loader, opt)
    v_x = Variable(x)
    # Y
    populate_xy(None, y_int, y_loader, opt)
    v_y_int = Variable(y_int, requires_grad=False)
    v_y = Variable(one_hot(y_int, opt))
    log_y = noise_log_y(v_y, temperature, opt.gpu_ids)
    # Z
    populate_z(z, opt)
    v_z = Variable(z)

    z_hat = net['H'](v_x)
    y_hat = net['F'](v_x)
    x_hat = net['G']( [log_y, v_z] )

    return (v_x, v_y_int, log_y, v_z, x_hat, y_hat, z_hat)

from util.meter import SegmentationMeter
## DEBUG
#FF = net['F'].state_dict()

def evaluation():
    xx = torch.FloatTensor(1, opt.input_nc, opt.heightSize, opt.widthSize)
    yy_int = torch.LongTensor(1, opt.heightSize, opt.widthSize)
    if len(opt.gpu_ids) > 0:
        xx = xx.cuda(opt.gpu_ids[0])
        yy_int = yy_int.cuda(opt.gpu_ids[0])

    ## DEBUG
    #for v1,v2 in zip(FF.values(), net['F'].state_dict().values()):
    #    if not torch.equal(v1,v2):
    #        print('*** netF MODIFIED!')
    #print('*** netF checked!')

    net['F'].eval()
    eval_stats = SegmentationMeter(n_class=opt.output_nc, ignore_index=opt.ignore_index)
    E_loss_CE = []

    start_time = time.time()
    val_loader_iter = iter(val_loader)
    for i in range(len(val_loader)):
        populate_xy(xx, yy_int, val_loader_iter, opt)
        v_x = Variable(xx, volatile=True)
        v_y_int = Variable(yy_int, volatile=True)
        y_hat = net['F'](v_x)
        E_loss_CE.append( CE(y_hat, v_y_int) )
        logits = y_hat.data.cpu().numpy()
        pred = logits.argmax(1) # NCHW -> NHW
        gt = yy_int.cpu().numpy()
        eval_stats.update_confmat(gt, pred)
    #print(x.mean(),logits.mean(),gt.mean()) # DEBUG: x,gt same; logits diff every other epoch

    print('EVAL ==> average CE = %.3f' % (sum(E_loss_CE).data[0] / len(val_loader)))
    eval_results = eval_stats.get_eval_results()
    msg = 'EVAL [%d images in %.2f sec] ==> %s\n' % \
                (len(val_loader), time.time()-start_time, eval_results[0])
    msg += 'Per-class IoU:\n'
    msg += ''.join(['%s: %.2f\n' % (cname,ciu)
                    for cname,ciu in zip(val_loader.dataset.label2name, eval_results[1])])
    print(msg)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s' % msg)

    if opt.update_F:
        net['F'].train()

from util.util import tensor2lab
def display_imgs(images, epoch, i, subset='train', do_save=0):
    for k, im in images.items():
        if 'y' in k:
            images[k] = tensor2lab(im, val_loader.dataset.label2color) # 3HW
        elif 'x' in k:
            images[k] = im[0] # 3HW
            d_mean = torch.FloatTensor(val_loader.dataset.mean).view(-1,1,1) # (3) -> (3,1,1)
            d_std = torch.FloatTensor(val_loader.dataset.std).view(-1,1,1)
            images[k] *= d_std
            images[k] += d_mean
            images[k] = images[k].mul(255).clamp(0,255).byte().numpy() # 3HW
    visualizer.display_current_results(images, epoch, i, subset=subset, do_save=do_save)

#########################################################################
# main loop
evaluation()
stats = {'D':0, 'G':0, 'P_CE':0, 'A_CE':0, 'P_L1':0, 'A_L1':0}
g_it = 0
for epoch in range(opt.start_epoch, opt.niter):

    # on begin epoch
    adjust_lr(epoch)
    print('netF.temperature = %.6f' % net['F'].temperature)
    epoch_start_time = time.time()

    for i in range(len(x_loader)):
        iter_start_time = time.time()

        if g_it % 100 == 0 and not opt.update_HG and opt.update_F:
            net['F'].temperature = np.maximum(TAU0*np.exp(-ANNEAL_RATE*g_it),MIN_TEMP)
        else:
            net['F'].temperature = MIN_TEMP

        if g_it < 25 or g_it % 500 == 0:
            D_ITERS, G_ITERS = 100, 1
        else:
            D_ITERS, G_ITERS = 5, 1

        if not opt.update_D:
            D_ITERS = 0

        # ---------------------------
        #        Optimize over D
        # ---------------------------
        for d_i in range(D_ITERS):
            v_x, v_y_int, log_y, v_z, x_hat, y_hat, z_hat = populate_xyz_hat(net['F'].temperature)

            E_q_D = net['D']( [v_x, y_hat.detach(), z_hat.detach()] ).mean()
            E_p_D = net['D']( [x_hat.detach(), log_y.detach(), v_z.view_as(z_hat)] ).mean()
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
                global g_it
                net['F'].zero_grad()
                net['G'].zero_grad()
                net['H'].zero_grad()
                sum(g_losses).backward()
                if opt.update_F:
                    optimizer['F'].step()
                if opt.update_HG:
                    optimizer['G'].step()
                    optimizer['H'].step()
                del g_losses[:]
                g_it += 1

            # paired X, Y
            populate_xy(x, y_int, paired_loader, opt)
            v_x = Variable(x)

            if opt.update_F:
                v_y_int = Variable(y_int, requires_grad=False)
                y_hat = net['F'](v_x) # log p(y | x)
                paired_loss_CE = CE(y_hat, v_y_int)
                g_losses.append( LAMBDA_CE * paired_loss_CE ) # CE
                stats['P_CE'] = paired_loss_CE.data[0]

            if opt.update_HG: # log p(x | y)
                z_hat = net['H'](v_x) # B x nz
                paired_loss_KL = KL(net['H'].mu, net['H'].logvar, num_pixs)
                g_losses.append( paired_loss_KL ) # KL
                stats['P_KL'] = paired_loss_KL.data[0]

                v_y = Variable(one_hot(y_int, opt))
                log_y = noise_log_y(v_y, net['F'].temperature, opt.gpu_ids)
                x_hat = net['G']( [log_y, z_hat.view_as(z)] ) # TODO: try v.z?
                paired_loss_L1 = L1(x_hat, v_x)
                g_losses.append( paired_loss_L1 ) # L1
                stats['P_L1'] = paired_loss_L1.data[0]

            update_FGH()

            # X, Y augmented
            # TODO: coeffs for g_losses
            if opt.update_HG: # log p(x)
                v_x, v_y_int, log_y, v_z, x_hat, y_hat, z_hat = populate_xyz_hat(net['F'].temperature)

                x_tilde = net['G']( [y_hat, z_hat.view_as(z)] ) # x -> y_hat, z_hat -> x_tilde
                aug_loss_L1 = L1(x_tilde, v_x)
                g_losses.append( aug_loss_L1 ) # L1

                y_tilde = net['F'](x_hat) # y,z -> x_hat -> y_tilde
                aug_loss_CE = CE(y_tilde, v_y_int)
                g_losses.append( aug_loss_CE ) # CE

                #z_tilde = net['H'](x_hat) # y,z -> x_hat -> z_tilde
                #aug_loss_KL = (v_z - net['H'].mu).pow(2).div(net['H'].logvar.exp().mul(2)).mean()
                #g_losses.append( aug_loss_KL ) # Gaussian

                stats['A_CE'] = aug_loss_CE.data[0]
                #stats['A_KL'] = aug_loss_KL.data[0]
                stats['A_L1'] = aug_loss_L1.data[0]

                if opt.update_D:
                    E_q_G = net['D']( [v_x, y_hat, z_hat] ).mean() # TODO: try D(x,y,z)
                    E_p_G = net['D']( [x_hat, v_y, v_z.view_as(z_hat)] ).mean()
                    g_loss = (E_q_G - E_p_G).pow(2)
                    g_losses.append( g_loss )

                    ##stats['E_q_G'] = E_q_G.data[0]
                    ##stats['E_p_G'] = E_p_G.data[0]
                    stats['G'] = g_loss.data[0]

                update_FGH()

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

            # visualization
            if opt.update_HG:
                images = {'x':v_x.data.cpu(), 'x_hat':x_hat.data.cpu(), 'x_tilde':x_tilde.data.cpu(),
                          'y':v_y_int.data.cpu().numpy(), 'y_hat':y_hat.data.cpu().numpy().argmax(1),
                          'y_tilde':y_tilde.data.cpu().numpy().argmax(1)}
            else:
                images = {'x':v_x.data.cpu(),
                          'y':v_y_int.data.cpu().numpy(), 'y_hat':y_hat.data.cpu().numpy().argmax(1)}
            display_imgs(images, epoch, i)

    # on end epoch
    print('===> End of epoch %d / %d \t Time Taken: %.2f sec\n' % \
                (epoch, opt.niter, time.time() - epoch_start_time))

    # evaluation & save
    if epoch % opt.save_every == 0:
        evaluation()
        for k in net.keys():
            if getattr(opt, 'update_%s' % k):
                weights_fpath = os.path.join(opt.checkpoints_dir, 'net%s_stage%s.pth' % (k,stage_str))
                torch.save(net[k].state_dict(), weights_fpath)
