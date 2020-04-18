import time
import datetime
import os
import sys
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import util.vutils as vutils
from args import parser
from collections import OrderedDict

from data.data_loader import CustomDatasetDataLoader, InfiniteDataLoader
from models.semantic_inductive_bias import SemanticInductiveBias
from util.util import tensor2lab
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#########################################################################
# options
opt = parser.parse_args()

# output directories
opt.out_dir = os.path.join(opt.out_dir, opt.dataset, opt.name,
                           'suprate%.3f_%d' % (opt.sup_portion, opt.seed))
os.makedirs(opt.out_dir, exist_ok=True)

# data_loaders
val_loader = CustomDatasetDataLoader(opt, istrain=False)
train_loader = CustomDatasetDataLoader(opt, istrain=True, suponly=False)
opt = train_loader.update_opt(opt)

## wrap with infinite loader
#train_loader = InfiniteDataLoader(train_loader)

# Visualizer
writer = SummaryWriter(opt.out_dir, purge_step=0, flush_secs=10)

# Logger
logger = vutils.get_logger(opt.out_dir, opt.name)

device = torch.device('cuda:%d' % opt.gpu)

# set rand seed
vutils.set_random_seed(opt.seed)

logger.info(opt)

#########################################################################
# algorithm functions

#-----------------------------------------------------------------------
def criterion(logits, target, img_hat, image, issup, coeff=0.01):
    if issup.any():
        ce = F.cross_entropy(logits[issup,...], target[issup], ignore_index=opt.ignore_index)
    else:
        ce = torch.tensor(0, dtype=logits.dtype, device=logits.device)
    l1 = F.l1_loss(img_hat, image)

    return ce + coeff * l1, ce.item(), l1.item()

#-----------------------------------------------------------------------
def get_scheduler(optimizer):
    if opt.lr_policy == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda epoch: (1 - epoch / (len(train_loader) * opt.epochs)) ** 0.9)
    elif opt.lr_policy == 'table':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 160], gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

#-----------------------------------------------------------------------
def evaluate(model, data_loader, writer, device, num_classes, epoch):
    model.eval()
    confmat = vutils.ConfusionMatrix(num_classes)
    metric_logger = vutils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 100, header):
            image, target = batch['A'].to(device), batch['B'].to(device)
            output = model.seg(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

        # visualization
        if epoch % opt.print_freq == 0:
            for i, batch in enumerate(data_loader):
                if i >= 5:
                    break
                image, target = batch['A'].to(device), batch['B'].to(device)
                logits, img_hat = model(image)

                if epoch == 0:
                    # image
                    imean = torch.tensor(data_loader.dataset.mean).view(-1,1,1)
                    istd = torch.tensor(data_loader.dataset.std).view(-1,1,1)
                    image = image[0].detach().cpu() * istd + imean
                    image = image.permute(1,2,0)
                    # label
                    label = target[0].detach()
                    label = tensor2lab(label, num_classes, data_loader.dataset.label2color)
                    # vis
                    fig, axeslist = plt.subplots(ncols=2, nrows=1)
                    axeslist[0].imshow(image)
                    axeslist[0].set_title('image')
                    axeslist[1].imshow(label)
                    axeslist[1].set_title('label')
                    writer.add_figure(f"gt/img-lab-{i}", fig, global_step=0)

                # image_hat
                image = img_hat[0].detach() * 0.5 + 0.5
                image = image.permute(1,2,0).cpu()
                fig = plt.figure()
                plt.imshow(image)
                writer.add_figure(f"image-pred/img-epoch{epoch}", fig, global_step=i)

                # logits
                label = logits[0].detach().argmax(0)
                label = tensor2lab(label, num_classes, data_loader.dataset.label2color)
                fig = plt.figure()
                plt.imshow(label)
                writer.add_figure(f"label-pred/lab-epoch{epoch}", fig, global_step=i)

    _, _, mIoU = confmat.compute()
    mIoU = mIoU.mean().item()
    writer.add_scalar("eval/mIoU", mIoU, global_step=epoch)

    return confmat, mIoU

#-----------------------------------------------------------------------
def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler,
                    writer, device, epoch, print_freq):
    model.train()
    metric_logger = vutils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', vutils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler.step()

    for idx, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target = batch['A'].to(device), batch['B'].to(device)
        logits, img_hat = model(image)

        loss, ce, kl = criterion(logits, target, img_hat, image, batch['issup'])

        optimizer.zero_grad()
        loss.backward()

        # DEBUG
        #logger.info('--------- DEBUG -----------')
        #for group in optimizer.param_groups:
        #    for p in group['params']:
        #        if p.grad is None: continue
        #        break
        #    logger.info(p.shape, p.grad.mean())
        #logger.info('--------- DEBUG -----------')

        optimizer.step()

        metric_logger.update(loss=loss.item(),
                             ce=ce, kl=kl,
                             lr=optimizer.param_groups[0]["lr"])

        writer.add_scalar("train/loss", loss.item(), global_step=epoch*len(data_loader) + idx)
        writer.add_scalar("train/ce", ce, global_step=epoch*len(data_loader) + idx)
        writer.add_scalar("train/kl", kl, global_step=epoch*len(data_loader) + idx)

#########################################################################
# main

#-----------------------------------------------------------------------
# model
model = SemanticInductiveBias(opt.output_nc, opt.x_drop).to(device)

#-----------------------------------------------------------------------
# optimizers
params_to_optimize = [
    {"params": [p for p in model.encoder.parameters() if p.requires_grad], 'lr': opt.lrs[0]},
    {"params": [p for p in model.decoder.parameters() if p.requires_grad], 'lr': opt.lrs[1]},
]

optimizer = torch.optim.Adam(params_to_optimize,
                             lr=opt.lrs[0], betas=(opt.beta1, 0.999))

lr_scheduler = get_scheduler(optimizer)

#-----------------------------------------------------------------------
# training
best_mIoU = 0
start_time = time.time()
for epoch in range(opt.epochs):
    train_one_epoch(model, criterion, optimizer, train_loader, lr_scheduler,
                    writer, device, epoch, opt.print_freq)
    confmat, mIoU = evaluate(model, val_loader, writer,
                             device, opt.output_nc, epoch)
    logger.info(confmat)

    if mIoU > best_mIoU:
        logger.info('==> Improved mIoU from %.3f --> %.3f\n' % (best_mIoU, mIoU))
        best_mIoU = mIoU
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'opt': opt
            },
            os.path.join(opt.out_dir, 'model_BEST.pth'))

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
logger.info('==> Training time {}'.format(total_time_str))

msg = 'mv {} {}'.format(os.path.join(opt.out_dir, 'model_BEST.pth'),
                        os.path.join(opt.out_dir, 'model_BEST{:.3f}.pth'.format(best_mIoU)))
logger.info(msg)
os.system(msg)
