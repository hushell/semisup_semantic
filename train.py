import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.trainer import CreateTrainer
from util.visualizer import Visualizer
from experiment_manager import ExperimentManager
import os
import torch

opt = TrainOptions().parse()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids # absolute ids
# to prevent opencv from initializing CUDA in workers
torch.randn(8).cuda()
os.environ['CUDA_VISIBLE_DEVICES'] = ''

opt.gpu_ids = range(0,len(opt.gpu_ids)) # new range starting from 0

######################################
# data_loaders, visualizer, trainer(models, optimizers)
opt.phase = 'val'
opt.isTrain = False
val_loader = CreateDataLoader(opt)
opt.phase = 'train'
opt.isTrain = True
train_loader = CreateDataLoader(opt)
visualizer = Visualizer(opt)
trainer = CreateTrainer(opt)

######################################
# exp_manager
expmgr = ExperimentManager(opt, trainer, visualizer, train_loader, val_loader)
if opt.continue_train:
    expmgr.resume(opt.which_epoch)

######################################
total_steps = 0 if not opt.continue_train else int(opt.which_epoch)*len(train_loader)
begin_epoch = 1 if not opt.continue_train else int(opt.which_epoch)+1

# on begin training
train_acc = expmgr.evaluation('train')
val_acc = expmgr.evaluation('val')
metrics = {'train_acc': train_acc, 'val_acc': val_acc, 'epoch': 0}
expmgr.update_history(metrics)

# main loop
for epoch in range(begin_epoch, opt.niter+opt.niter_decay+1):
    # on begin epoch
    trainer.update_learning_rate(epoch)
    epoch_start_time = time.time()

    for i, data in enumerate(train_loader):
        total_steps += opt.batchSize

        # gradient step
        iter_start_time = time.time()
        trainer.set_input(data)
        trainer.optimize_parameters()
        t = (time.time() - iter_start_time) / opt.batchSize

        # plot images
        if total_steps % opt.display_freq == 0:
            expmgr.plot_current_images(epoch, i, do_save=False)

        # plot metrics
        if total_steps % opt.print_freq == 0:
            expmgr.print_plot_current_losses(epoch, total_steps, t)

    # on end epoch
    expmgr.plot_current_images(epoch, i, do_save=True)
    msg = '===> End of epoch %d / %d \t Time Taken: %.2f sec\n' % \
                (epoch, opt.niter+opt.niter_decay, time.time() - epoch_start_time)

    if epoch % opt.save_epoch_freq == 0:
        msg += '===> Saving the model at the end of epoch %d, total_iters %d\n' % (epoch, total_steps)
        expmgr.save_weights(epoch)
        expmgr.save_optimizer(epoch)

        train_acc = expmgr.evaluation('train')
        val_acc = expmgr.evaluation('val')
        metrics = {'train_acc': train_acc, 'val_acc': val_acc, 'epoch': epoch, 'lr': trainer.old_lr}
        expmgr.update_history(metrics)
        expmgr.save_history()

    print(msg)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s' % msg)

# on end training
expmgr.save_history()
