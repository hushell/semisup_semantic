import time
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import util.vutils as vutils
from args import parser
from collections import OrderedDict

from data.data_loader import CustomDatasetDataLoader, InfiniteDataLoader
#from util.visualizer import Visualizer
from models.semantic_inductive_bias import SemanticInductiveBias


#########################################################################
# options
opt = parser.parse_args()
opt.name += '_%s/%d' % (opt.dataset, opt.manual_seed)
device = torch.device('cuda:%d' % opt.gpu)

# data_loaders
val_loader = CustomDatasetDataLoader(opt, istrain=False)
train_loader = CustomDatasetDataLoader(opt, istrain=True, issup=False)
opt = train_loader.update_opt(opt)

## wrap with infinite loader
#train_loader = InfiniteDataLoader(train_loader)

## Visualizer
#visualizer = Visualizer(opt)

print(opt)


#########################################################################
# algorithm functions

def criterion(logits, target, img_hat, image, issup, coeff=0.01):
    if issup.any():
        ce = F.cross_entropy(logits[issup,...], target[issup], ignore_index=opt.ignore_index)
    else:
        ce = torch.tensor(0, dtype=logits.dtype, device=logits.device)
    l1 = F.l1_loss(img_hat, image)

    # DEBUG
    if torch.isnan(ce) or torch.isnan(l1):
        import ipdb; ipdb.set_trace()

    return ce + coeff * l1, ce, l1

#-----------------------------------------------------------------------
def get_scheduler(optimizer):
    # TODO
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch): # decay to 0 starting from epoch=niter_decay
            lr_l = 1.0 - max(0, epoch+1+opt.start_epoch-opt.niter+opt.niter_decay) / float(opt.niter_decay+1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

#-----------------------------------------------------------------------
from util.util import tensor2lab
def display_imgs(images, epoch, i, subset='train', do_save=0):
    # TODO
    if opt.display_id <= 0 and opt.no_html:
        return

    for k, im in images.items():
        if 'B' in k:
            images[k] = tensor2lab(im, n_labs=opt.output_nc) # 3HW
        elif 'A' in k:
            images[k] = im[0] # 3HW
            d_mean = torch.FloatTensor(val_loader.dataset.mean).view(-1,1,1) # (3) -> (3,1,1)
            d_std = torch.FloatTensor(val_loader.dataset.std).view(-1,1,1)
            images[k] *= d_std
            images[k] += d_mean
            images[k] = images[k].mul(255).clamp(0,255).byte().numpy() # 3HW
        elif 'mask' in k:
            images[k] = tensor2lab(im, n_labs=2) # 3HW
    visualizer.display_current_results(images, epoch, i, subset=subset, do_save=do_save)

#-----------------------------------------------------------------------
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = vutils.ConfusionMatrix(num_classes)
    metric_logger = vutils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 100, header):
            image, target = batch['A'].to(device), batch['B'].to(device)
            output = model.seg(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())

            # TODO visualization

        confmat.reduce_from_all_processes()

    return confmat

#-----------------------------------------------------------------------
def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = vutils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', vutils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        image, target = batch['A'].to(device), batch['B'].to(device)
        logits, img_hat = model(image)

        loss, ce, kl = criterion(logits, target, img_hat, image, batch['issup'])

        optimizer.zero_grad()
        loss.backward()

        # DEBUG
        #print('--------- DEBUG -----------')
        #for group in optimizer.param_groups:
        #    for p in group['params']:
        #        if p.grad is None: continue
        #        break
        #    print(p.shape, p.grad.mean())
        #print('--------- DEBUG -----------')

        optimizer.step()

        #lr_scheduler.step()

        metric_logger.update(loss=loss.item(),
                             ce=ce, kl=kl,
                             lr=optimizer.param_groups[0]["lr"])


#########################################################################
# main

#-----------------------------------------------------------------------
# model
model = SemanticInductiveBias(opt.output_nc).to(device)

#-----------------------------------------------------------------------
# optimizers
params_to_optimize = [
    {"params": [p for p in model.encoder.parameters() if p.requires_grad], 'lr': opt.lrs[0]},
    {"params": [p for p in model.decoder.parameters() if p.requires_grad], 'lr': opt.lrs[1]},
]

optimizer = torch.optim.Adam(params_to_optimize,
                             lr=opt.lrs[0], betas=(opt.beta1, 0.999))

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lambda x: (1 - x / (len(train_loader) * opt.epochs)) ** 0.9)

#-----------------------------------------------------------------------
# training
start_time = time.time()
for epoch in range(opt.epochs):
    train_one_epoch(model, criterion, optimizer, train_loader, lr_scheduler, device, epoch, opt.print_freq)
    confmat = evaluate(model, val_loader, device=device, num_classes=opt.output_nc)
    print(confmat)
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'opt': opt
        },
        os.path.join(opt.output_dir, 'model_{}.pth'.format(opt.name)))

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))




