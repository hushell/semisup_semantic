import time
import datetime
import os
import sys
import torch
import torch.nn.functional as F
import util.vutils as vutils

from tensorboardX import SummaryWriter
from args import parser
from data.data_loader import CustomDatasetDataLoader, InfiniteDataLoader
from models.baseline import Baseline
from models.semantic_reconstruct import SemanticReconstruct
from models.semantic_consistency import SemanticConsistency

import matplotlib
matplotlib.use("Agg")


#########################################################################
# options
opt = parser.parse_args()

# output directories
opt.out_dir = os.path.join(opt.out_dir, opt.dataset, opt.name,
                           'suprate%.3f_droprate%.2f_seed%d' % (opt.sup_portion, opt.x_drop, opt.seed))
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
# algorithm functions (#TODO: class Algorithm)

#-----------------------------------------------------------------------
def criterion(logits, target, issup, aux_loss=None):
    if issup.any():
        ce = F.cross_entropy(logits[issup,...], target[issup], ignore_index=opt.ignore_index)
    else:
        ce = torch.tensor(0, dtype=logits.dtype, device=logits.device)

    if aux_loss is None:
        loss = ce
    else:
        loss = ce + opt.coeff * aux_loss

    return loss, ce

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
            model.vis(data_loader, writer, epoch, num_classes, device)

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
        logits, aux_loss = model(image)

        loss, ce = criterion(logits, target, batch['issup'], aux_loss)

        if loss.grad_fn is not None: # forwarded successfully
            optimizer.zero_grad()
            loss.backward()
            writer.add_scalar("train/loss", loss.item(), global_step=epoch*len(data_loader) + idx)

            # DEBUG NAN
            #logger.info('--------- DEBUG -----------')
            #for group in optimizer.param_groups:
            #    for p in group['params']:
            #        if p.grad is None: continue
            #        break
            #    logger.info(p.shape, p.grad.mean())
            #logger.info('--------- DEBUG -----------')

            optimizer.step()

        if ce.grad_fn is not None:
            writer.add_scalar("train/ce", ce.item(), global_step=epoch*len(data_loader) + idx)

        if aux_loss is not None:
            writer.add_scalar("train/aux_loss", aux_loss.item(), global_step=epoch*len(data_loader) + idx)
            metric_logger.update(loss=loss.item(),
                                 ce=ce.item(), aux_loss=aux_loss.item(),
                                 lr=optimizer.param_groups[0]["lr"])
        else:
            metric_logger.update(loss=loss.item(),
                                 lr=optimizer.param_groups[0]["lr"])


#########################################################################
# main

#-----------------------------------------------------------------------
# model
if opt.model == 'baseline':
    model = Baseline(opt.output_nc).to(device)
elif opt.model == 'reconstruct':
    model = SemanticReconstruct(opt.output_nc, opt.x_drop).to(device)
elif opt.model == 'consistency':
    model = SemanticConsistency(opt.output_nc, 256, opt.x_drop).to(device)

#-----------------------------------------------------------------------
# optimizers
params = model.params_to_optimize(opt.lrs)

optimizer = torch.optim.Adam(params, lr=opt.lrs[0], betas=(opt.beta1, 0.999))

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
    logger.info('====== Epoch %d ======' % epoch)
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
