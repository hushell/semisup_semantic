from __future__ import print_function
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.trainer import CreateTrainer
from util.visualizer import Visualizer
from experiment_manager import ExperimentManager
from util.meter import SegmentationMeter
import os
import torch

opt = TrainOptions().parse()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids # absolute ids
# to prevent opencv from initializing CUDA in workers
if len(opt.gpu_ids) > 0:
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
opt = train_loader.update_opt(opt)

visualizer = Visualizer(opt)
trainer = CreateTrainer(opt)

# exp_manager
expmgr = ExperimentManager(opt, trainer, visualizer, train_loader, val_loader)

######################################
def eval_ckpts(data_loader, subset=1, desc='train_LABELED', tag=0):
    for which_epoch in range(100, 1001, 100):
        n_samples = len(data_loader) if subset > 1 else (data_loader.dataset.unsup.sum() if subset == 0 else len(data_loader)-data_loader.dataset.unsup.sum())
        cum_losses = {'D_A':0.0, 'G_A-G_B-GAN':0.0, 'G_A-G_B-L1':0.0, 'G_A-CE':0.0}
        eval_stats = SegmentationMeter(n_class=opt.output_nc, ignore_index=opt.ignore_index)

        expmgr.resume(which_epoch)

        jj = 0
        start_time = time.time()
        for i, data in enumerate(data_loader):
            if data.has_key('unsup'):
                if subset == data['unsup'][0]: # subset=1 sup; subset=0 unsup, subset=2 all
                    continue
            print('%d' % i, end=' ')

            # forward
            trainer.set_input(data)
            trainer.test()
            trainer.rec_A = trainer.models['G_B'].forward(trainer.fake_B) # G_B(G_A(A))

            # loss D_A
            trainer.optimizers['D_A'].zero_grad()
            trainer.backward_D_A(trainer.rec_A, True)
            # losses G_A and G_B
            trainer.optimizers['G_A'].zero_grad()
            trainer.optimizers['G_B'].zero_grad()
            trainer.backward_G_AB(True)

            # cumulate losses
            losses = trainer.get_current_losses()
            for k,v in losses.iteritems():
                if cum_losses.has_key(k):
                    cum_losses[k] += v

            # eval
            pred, gt = trainer.get_eval_pair()
            eval_stats.update_confmat(gt, pred)

            # save images
            if jj < 10:
                do_save = 2 if jj < 3 else 1
                expmgr.plot_current_images(which_epoch, i+tag, subset='val', do_save=do_save)
            jj += 1

        # print results
        eval_results = eval_stats.get_eval_results()
        msg = '\n----------------------epoch %d ------------------------\n' % which_epoch
        msg += '==> %s Results [%d images] \t Time Taken: %.2f sec: %s\n' % \
                    (desc, n_samples, time.time()-start_time, eval_results[0])
        msg += '==> Per-class IoU:\n'
        msg += ''.join(['%s: %.2f\n' % (cname,ciu)
                        for cname,ciu in zip(data_loader.dataset.label2name, eval_results[1])])
        msg += '==> Losses:\n'
        msg += ''.join(['%s: %.2f\n' % (cname, ciu / n_samples)
                        for cname,ciu in cum_losses.items()])
        print(msg)
        with open(visualizer.log_name, "a") as log_file:
            log_file.write('%s' % msg)

        # create/update html
        visualizer.save_webpage('val')

######################################
# losses (train labeld)
eval_ckpts(train_loader, subset=1, desc='TRAIN_LABELED', tag=0)

# losses (train unlabeld)
eval_ckpts(train_loader, subset=0, desc='TRAIN_UNLABELED', tag=10000)

# losses (val)
eval_ckpts(val_loader, subset=2, desc='TEST', tag=20000)

