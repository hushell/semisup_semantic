import os
import time
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import util.util as util
from util.meter import SegmentationMeter


class ExperimentManager():
    def __init__(self, opt, trainer, visualizer, train_loader, val_loader=None, test_loader=None):
        # components
        self.opt = opt
        self.trainer = trainer
        self.visualizer = visualizer
        self.metrics_history = pd.DataFrame(columns=['train_acc', 'val_acc', 'epoch', 'lr'])
        self.data_loader = dict()
        self.data_loader['train'] = train_loader
        self.data_loader['val'] = val_loader
        self.data_loader['test'] = test_loader

        # paths
        self.ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.results_dir = os.path.join(self.ckpt_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.history_fpath = os.path.join(self.ckpt_dir, 'history.xlsx')

    def resume(self, epoch=None):
        self.load_history()
        if epoch is None or epoch == 'latest':
            epoch = self.metrics_history.loc[self.metrics_history.last_valid_index(), 'epoch']

        self.load_weights(epoch)
        if self.opt.isTrain:
            self.load_optimizer(epoch)

    def update_history(self, metrics):
        ''' metrics is a dict with keys a subset of pd.columns
        '''
        self.metrics_history = self.metrics_history.append(metrics, ignore_index=True)

    def save_history(self):
        self.metrics_history.to_excel(self.history_fpath, sheet_name='metrics_history')

    def load_history(self):
        assert(os.path.exists(self.history_fpath))
        self.metrics_history = pd.read_excel(self.history_fpath, sheet_name='metrics_history')

    def save_weights(self, epoch_label):
        for network_label in self.trainer.models.iterkeys():
            weights_fname = '%s_net_%s.pth' % (network_label, epoch_label)
            weights_fpath = os.path.join(self.ckpt_dir, weights_fname)
            torch.save(self.trainer.models[network_label].state_dict(), weights_fpath)

    def load_weights(self, epoch_label):
        for network_label in self.trainer.models.iterkeys():
            weights_fname = '%s_net_%s.pth' % (network_label, epoch_label)
            weights_fpath = os.path.join(self.ckpt_dir, weights_fname)
            assert(os.path.exists(weights_fpath))
            state = torch.load(weights_fpath)
            self.trainer.models[network_label].load_state_dict(state)

    def save_optimizer(self, epoch_label):
        for network_label in self.trainer.optimizers.iterkeys():
            optim_fname = '%s_opt_%s.pth' % (network_label, epoch_label)
            optim_fpath = os.path.join(self.ckpt_dir, optim_fname)
            torch.save(self.trainer.optimizers[network_label].state_dict(), optim_fpath)

    def load_optimizer(self, epoch_label):
        for network_label in self.trainer.optimizers.iterkeys():
            optim_fname = '%s_opt_%s.pth' % (network_label, epoch_label)
            optim_fpath = os.path.join(self.ckpt_dir, optim_fname)
            assert(os.path.exists(optim_fpath))
            optim = torch.load(optim_fpath)
            self.trainer.optimizers[network_label].load_state_dict(optim)

        self.trainer.old_lr = self.trainer.optimizers['G_A'].param_groups[0]['lr'] / self.trainer.lr_coeffs['G_A']

    def plot_current_images(self, epoch, i, subset='train', do_save=0):
        images = self.trainer.get_current_visuals()
        for k, im in images.items():
            if 'B' in k:
                images[k] = util.tensor2lab(im, self.data_loader['train'].dataset.label2color)
            else:
                images[k] = util.tensor2im(im, imtype=np.float32)
                d_mean = np.array(self.data_loader['train'].dataset.mean)
                d_std = np.array(self.data_loader['train'].dataset.std)
                d_mean = d_mean[:,np.newaxis,np.newaxis]
                d_std = d_std[:,np.newaxis,np.newaxis]
                images[k] *= d_std
                images[k] += d_mean
                images[k] *= 255.0
                images[k] = images[k].astype(np.uint8)
        self.visualizer.display_current_results(images, epoch, i, subset=subset, do_save=do_save)

    def print_plot_current_losses(self, epoch, total_i, t):
        losses = self.trainer.get_current_losses()
        self.visualizer.print_current_metrics(epoch, total_i, t, losses)
        self.visualizer.plot_current_metrics(losses, total_i)

    def evaluation(self, phase='train'):
        self.trainer.train(mode=False)

        eval_stats = SegmentationMeter(n_class=self.opt.output_nc, ignore_index=self.opt.ignore_index)
        start_time = time.time()
        for i,data in enumerate(self.data_loader[phase]):
            self.trainer.set_input(data)
            self.trainer.test(phase)
            pred, gt = self.trainer.get_eval_pair()
            eval_stats.update_confmat(gt, pred)

            if phase is 'test':
                do_save = 2 if i < 10 else 1
            elif phase is 'val':
                do_save = 2 if i < 3 else 0
            else:
                do_save = 0
            self.plot_current_images(9999, i, subset=phase, do_save=do_save)

        eval_results = eval_stats.get_eval_results()
        msg = '==> %s Results [%d images] \t Time Taken: %.2f sec: %s\n' % \
                    (phase, len(self.data_loader[phase]), time.time()-start_time, eval_results[0])
        msg += 'Per-class IoU:\n'
        msg += ''.join(['%s: %.2f\n' % (cname,ciu)
                        for cname,ciu in zip(self.data_loader['train'].dataset.label2name, eval_results[1])])
        print(msg)
        with open(self.visualizer.log_name, "a") as log_file:
            log_file.write('%s' % msg)

        if phase is 'test':
            self.visualizer.save_webpage(prefix='test')

        self.trainer.train(mode=True)
        return eval_results[0]['Mean IoU']

    def on_begin_epoch(self, epoch):
        self.trainer.on_begin_epoch(epoch)
        print('==> lambda_A = %f' % self.trainer.opt.lambda_A)
        print('==> lambda_B = %f' % self.trainer.opt.lambda_B)

    def on_end_epoch(self, epoch):
        self.trainer.on_end_epoch(epoch)

