import time
import os
import torch
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
from args import parser

#########################################################################
# options
opt = parser.parse_args()
opt.name += '_%s/lr%s_lb%s' % (opt.dataset, opt.lrs, opt.lambdas)

print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

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

net =dict()
net['X2Z'] = FX2Z(opt)
net['Z2Y'] = FZ2Y(opt, n_blocks=2)
net['ZY2X'] = GZ2X(opt) # TODO: ZY2X to Z2X
net['D'] = NLayerDiscriminator(opt)

for k in net.keys():
    if len(opt.gpu_ids) > 0:
        net[k].cuda(opt.gpu_ids[0])

    # init
    net[k].apply(weights_init)

    # load if found saved weights
    weights_fpath = os.path.join(opt.checkpoints_dir, opt.name, 'net%s.pth' % (k))
    if os.path.exists(weights_fpath):
        print('Load net[%s] from %s' % (k, weights_fpath))
        net[k].load_state_dict(torch.load(weights_fpath, map_location=lambda storage, loc: storage))

    # train or freeze
    if opt.updates[k] == 2:
        print('Training net[%s]' % k)
        net[k].train()
        for param in net[k].parameters():
            param.requires_grad = True
    else:
        print('Freezing or disabling net[%s]' % k)
        net[k].eval() # NOTE: to disable batchnorm updates
        for param in net[k].parameters():
            param.requires_grad = False

#########################################################################
# optimizers
from torch.optim import lr_scheduler
def get_scheduler(optimizer):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch): # decay to 0 starting from epoch=niter_decay
            lr_l = 1.0 - max(0, epoch+1+opt.start_epoch-opt.niter+opt.niter_decay) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

optimizers = dict()
schedulers = dict()
for k in net.keys():
    if opt.updates[k] == 2:
        print('Creating optimizer for net[%s]' % k)
        optimizers[k] = optim.Adam(net[k].parameters(), lr=opt.lrs[k], betas=(opt.beta1, 0.999))
        schedulers[k] = get_scheduler(optimizers[k])

def adjust_lr(epoch):
    print('===> Start of epoch %d / %d' % (epoch, opt.niter))
    for k,sch in schedulers.items():
        sch.step()
        lr = optimizers[k].param_groups[0]['lr']
        print('%s: learning rate = %.7f' % (k,lr))

# losses
CE, L1 = create_losses(opt)

#########################################################################
# variables
t_x = torch.FloatTensor(opt.batchSize, opt.input_nc, opt.heightSize, opt.widthSize)
t_y_int = torch.LongTensor(opt.batchSize, opt.heightSize, opt.widthSize)

heightSize = val_loader.dataset.heightSize
widthSize = val_loader.dataset.widthSize
xx = torch.FloatTensor(1, opt.input_nc, heightSize, widthSize)
yy_int = torch.LongTensor(1, heightSize, widthSize)

if len(opt.gpu_ids) > 0:
    t_x = t_x.cuda(opt.gpu_ids[0])
    t_y_int = t_y_int.cuda(opt.gpu_ids[0])
    xx = xx.cuda(opt.gpu_ids[0])
    yy_int = yy_int.cuda(opt.gpu_ids[0])

#-----------------------------------------------------------------------
def generate_var_x(tensor=t_x, loader=x_loader):
    populate_xy(tensor, None, loader, opt)
    v_x = Variable(tensor)
    return v_x

def generate_var_y(tensor=t_y_int, loader=y_loader, temperature=0.0):
    populate_xy(None, tensor, loader, opt)
    v_y_int = Variable(tensor, requires_grad=False)
    v_y = Variable(one_hot(tensor, opt))
    log_y = noise_log_y(v_y, temperature, opt.gpu_ids)
    return (v_y_int, log_y)

def generate_var_xy(tensors=[t_x,t_y_int], loader=paired_loader, volatile=False, temperature=0.0):
    populate_xy(tensors[0], tensors[1], loader, opt)
    v_x = Variable(tensors[0], volatile=volatile)
    v_y_int = Variable(tensors[1], requires_grad=False, volatile=volatile)
    v_y = Variable(one_hot(tensors[1], opt), volatile=volatile)
    log_y = noise_log_y(v_y, temperature, opt.gpu_ids)
    return (v_x, v_y_int, log_y)

def Y_step(v_x, v_y_int):
    z_hat = net['X2Z'](v_x)
    y_hat = net['Z2Y'](z_hat)
    loss = None if v_y_int is None else CE(y_hat, v_y_int)
    return (z_hat, y_hat, loss)

def X_step(z_hat, v_x_drop, v_x):
    x_inf = net['ZY2X'](z_hat, v_x_drop)
    return (x_inf, L1(x_inf, v_x))

def update_FHG(g_losses):
    global g_it
    net['X2Z'].zero_grad()
    net['Z2Y'].zero_grad()
    net['ZY2X'].zero_grad()
    sum(g_losses).backward()
    if opt.updates['X2Z'] == 2: optimizers['X2Z'].step()
    if opt.updates['Z2Y'] == 2: optimizers['Z2Y'].step()
    if opt.updates['ZY2X'] == 2: optimizers['ZY2X'].step()
    del g_losses[:] # empty affects outside
    g_it += 1

def D_step(v_x):
    z_hat = net['X2Z'](v_x)
    #y_hat = net['Z2Y'](z_hat)
    v_x_drop,_ = mask_drop(v_x)
    x_hat = net['ZY2X'](z_hat, v_x_drop)

    E_q_D = net['D']( x_hat.detach() ).mean()
    E_p_D = net['D']( v_x ).mean()
    d_loss = -lambda_x * (E_q_D - E_p_D).pow(2)

    optimizers['D'].zero_grad()
    d_loss.backward()
    optimizers['D'].step()
    # clamp parameters to a cube
    for p in net['D'].parameters():
        p.data.clamp_(CLAMP_LOW, CLAMP_UPP)

    return d_loss

from torch.nn.functional import dropout
def mask_drop(v_x):
    #v_x0 = dropout(v_x[:,0,:,:], p=opt.x_drop, training=True)
    #mask = v_x0.le(0).unsqueeze_(1).detach()
    #v_x_drop = v_x.masked_fill(mask, 0)

    xrand = Variable(torch.rand(v_x.size(0), 1, v_x.size(2) * v_x.size(3)))
    xrand = xrand.cuda() if v_x.is_cuda else xrand
    _, indices = xrand.topk(k=int((1-opt.x_drop) * v_x.size(2) * v_x.size(3)), dim=2)
    mask = Variable(torch.zeros(xrand.size()))
    mask = mask.cuda() if v_x.is_cuda else mask
    mask.scatter_(2, indices, 1.0)
    mask = mask.view(v_x.size(0), 1, v_x.size(2), v_x.size(3))
    v_x_drop = v_x.mul(mask)
    return v_x_drop, mask

#-----------------------------------------------------------------------
from util.meter import SegmentationMeter

def evaluation(epoch, do_G=False, subset='train'):
    net['X2Z'].eval()
    net['Z2Y'].eval()
    if do_G: net['ZY2X'].eval()

    eval_stats = SegmentationMeter(n_class=opt.output_nc, ignore_index=opt.ignore_index)
    E_loss = [[0], [0], [0]] # CE, A_L1, P_L1

    lval = len(val_loader)
    start_time = time.time()
    val_loader_iter = iter(val_loader)
    for i in range(lval):
        v_x, v_y_int, log_y = generate_var_xy([xx, yy_int], val_loader_iter, True)

        z_hat, y_hat, P_CE = Y_step(v_x, v_y_int)
        E_loss[0].append( P_CE.data[0] )

        logits = y_hat.data.cpu().numpy()
        pred = logits.argmax(1) # NCHW -> NHW
        gt = yy_int.cpu().numpy()
        eval_stats.update_confmat(gt, pred)

        if do_G:
            v_x_drop, mask = mask_drop(v_x)
            x_hat, A_L1 = X_step(z_hat, v_x_drop, v_x)
            #x_tilde, P_L1 = X_step(z_hat, log_y, v_x)
            E_loss[1].append( A_L1.data[0] )
            #E_loss[2].append( P_L1.data[0] )
            E_loss[2].append( A_L1.data[0] )

        # visualization
        if i % 200 == 0:
            images = {'T_x':v_x.data.cpu(),
                      'T_y':v_y_int.data.cpu().numpy(), 'T_y_hat':y_hat.data.cpu().numpy().argmax(1),
                      'T_z_hat':z_hat.data.cpu().numpy().argmax(1)}
            if do_G:
                images['T_x_drop'] = v_x_drop.data.cpu()
                images['T_x_hat']  = x_hat.data.cpu()
                #images['T_mask']   = mask.data.cpu().numpy().squeeze(1)
                #images['T_x_tilde'] = x_tilde.data.cpu()
            display_imgs(images, epoch, i, subset=subset, do_save=2)

    print('EVAL at epoch %d ==> CE = %.3f, A_L1 = %.3f, P_L1 = %.3f' % \
            (epoch, sum(E_loss[0])/lval, sum(E_loss[1])/lval, sum(E_loss[2])/lval))
    eval_results = eval_stats.get_eval_results()
    msg = 'EVAL at epoch %d [%d images in %.2f sec] ==> %s\n' % \
            (epoch, lval, time.time()-start_time, eval_results[0])
    msg += 'Per-class IoU:\n'
    msg += ''.join(['%s: %.2f\n' % (cname,ciu)
                    for cname,ciu in zip(val_loader.dataset.label2name, eval_results[1])])
    print(msg)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s' % msg)

    if opt.updates['X2Z'] == 2:
        net['X2Z'].train()
    if opt.updates['Z2Y'] == 2:
        net['Z2Y'].train()
    if opt.updates['ZY2X'] == 2:
        net['ZY2X'].train()

    return eval_results[0]['Mean IoU']

#-----------------------------------------------------------------------
from util.util import tensor2lab
def display_imgs(images, epoch, i, subset='train', do_save=0):
    if opt.display_id <= 0 and opt.no_html:
        return

    for k, im in images.items():
        if 'y' in k:
            images[k] = tensor2lab(im, n_labs=opt.output_nc) # 3HW
        elif 'z' in k:
            images[k] = tensor2lab(im, n_labs=opt.z_nc) # 3HW
        elif 'x' in k:
            images[k] = im[0] # 3HW
            d_mean = torch.FloatTensor(val_loader.dataset.mean).view(-1,1,1) # (3) -> (3,1,1)
            d_std = torch.FloatTensor(val_loader.dataset.std).view(-1,1,1)
            images[k] *= d_std
            images[k] += d_mean
            images[k] = images[k].mul(255).clamp(0,255).byte().numpy() # 3HW
        elif 'mask' in k:
            images[k] = tensor2lab(im, n_labs=2) # 3HW
    visualizer.display_current_results(images, epoch, i, subset=subset, do_save=do_save)

#########################################################################
mIoU = evaluation(0, do_G=True)
stats = {'D':0, 'G':0, 'P_CE':0, 'A_CE':0, 'P_L1':0, 'A_L1':0}
g_it = 0

# main loop
for epoch in range(opt.start_epoch, opt.niter):
    # on begin epoch
    adjust_lr(epoch)
    epoch_start_time = time.time()

    for i in range(len(x_loader)):
        tt0 = time.time()
        if g_it < 25 or g_it % 500 == 0:
            D_ITERS, G_ITERS = 20, 1 # 100, 1
        else:
            D_ITERS, G_ITERS = 2, 1 # 5, 1
        if opt.updates['D'] != 2:
            D_ITERS = 0

        # -------------------------------------------------------------------
        #        Optimize over D
        # -------------------------------------------------------------------
        # TODO: D(x, y_hat) v.s. D(x_hat, log_y): pix2pixHD or sngan_projection
        for d_i in range(D_ITERS):
            v_x = generate_var_x()
            d_loss = D_step(v_x)
            stats['D'] = -d_loss.data[0]

        # -------------------------------------------------------------------
        #        Optimize over FX2Z, HZ2Y, GZY2X
        # -------------------------------------------------------------------
        for g_i in range(G_ITERS):
            tt1 = time.time()
            images = {}

            # -------------------------------------------------------------------
            # paired X, Y
            g_losses = []
            v_x, v_y_int, log_y = generate_var_xy()

            # log p(y | x)
            z_hat, y_hat, P_CE = Y_step(v_x, v_y_int)
            g_losses.append( lambda_y * P_CE ) # CE
            stats['P_CE'] = P_CE.data[0]

            # visdom
            images['xP'] = v_x.data.cpu()
            images['yP'] = v_y_int.data.cpu().numpy()
            images['yP_hat'] = y_hat.data.cpu().numpy().argmax(1)

            # log p(x | y)
            if opt.updates['ZY2X'] == 2:
                v_x_drop,_ = mask_drop(v_x)
                x_tilde, P_L1 = X_step(z_hat, v_x_drop, v_x)
                g_losses.append( lambda_x * P_L1 ) # L1
                stats['P_L1'] = P_L1.data[0]

                # visdom
                images['xP_tilde'] = x_tilde.data.cpu()

            update_FHG(g_losses)

            tt2 = time.time()
            # -------------------------------------------------------------------
            # X augmented
            # TODO: coeffs for g_losses: L1 and (D_R - D_F)^2 should be weighted

            # log p(x)
            if opt.updates['ZY2X'] > 0 and opt.updates['X2Z'] > 0:
                g_losses = []
                v_x = generate_var_x()
                v_x_drop,_ = mask_drop(v_x)

                z_hat, y_hat, A_CE = Y_step(v_x, None)
                x_hat, A_L1 = X_step(z_hat, v_x_drop, v_x)

                g_losses.append( lambda_x * A_L1 ) # L1
                stats['A_L1'] = A_L1.data[0]

                if opt.updates['D'] > 0:
                    E_q_G = net['D']( x_hat ).mean()
                    E_p_G = net['D']( v_x ).mean()
                    g_loss = (E_q_G - E_p_G).pow(2)
                    g_losses.append( lambda_x * g_loss )
                    stats['G'] = g_loss.data[0]

                update_FHG(g_losses)

                # visdom
                images['x'] = v_x.data.cpu()
                images['y_hat'] = y_hat.data.cpu().numpy().argmax(1)
                images['x_hat'] = x_hat.data.cpu()

        # time spent per sample
        tt3 = time.time()

        # -------------------------------------------------------------------
        # print & plot
        if i % (len(x_loader)/3) == 0:
            print('[{epoch}/{nepoch}][{iter}/{niter}] in {t:.3f}s ({t01:.3f},{t12:.3f},{t23:.3f}) '
                  'D/G: {D:.3f}/{G:.3f} '
                  'P_CE/A_CE: {P_CE:.3f}/{A_CE:.3f} '
                  'P_L1/A_L1: {P_L1:.3f}/{A_L1:.3f} '
                  ''.format(epoch=epoch, nepoch=opt.niter,
                            iter=i, niter=len(x_loader),
                            t=tt3-tt0, t01=tt1-tt0, t12=tt2-tt1, t23=tt3-tt2,
                            **stats))
            display_imgs(images, epoch, i)

    # -------------------------------------------------------------------
    # on end epoch
    print('===> End of epoch %d / %d \t Time Taken: %.2f sec\n' % \
            (epoch, opt.niter, time.time() - epoch_start_time))

    # evaluation & save (best only)
    if epoch % opt.save_every == 0:
        visualizer.save_webpage(prefix='train') # visualizer maintains a img_dict to be saved in webpage
        temp_mIoU = evaluation(epoch, do_G=True) # figures are saved in train folder
        if temp_mIoU >= mIoU:
            mIoU = temp_mIoU
            for k in net.keys():
                if opt.updates[k] == 2:
                    weights_fpath = os.path.join(opt.checkpoints_dir, opt.name, 'net%s.pth' % (k))
                    torch.save(net[k].state_dict(), weights_fpath)

