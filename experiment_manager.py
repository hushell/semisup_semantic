import os
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
            torch.save(self.trainer.models[network_label], weights_fpath)

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

    def plot_current_images(self, epoch, i, do_save=False):
        images = self.trainer.get_current_visuals()
        for k, im in images.items():
            if 'B' in k:
                images[k] = util.tensor2lab(im, self.data_loader['train'].dataset.label2color)
            else:
                images[k] = util.tensor2im(im)
        self.visualizer.display_current_results(images, epoch, i, do_save=do_save)

    def print_plot_current_losses(self, epoch, total_i, t):
        losses = self.trainer.get_current_losses()
        self.visualizer.print_current_metrics(epoch, total_i, t, losses)
        self.visualizer.plot_current_metrics(losses, total_i)

    def evaluation(phase='train'):
        eval_stats = SegmentationMeter(n_class=opt.output_nc)
        for data in self.data_loader[phase]:
            self.trainer.set_input(data)
            self.trainer.test()
            pred, gt = self.trainer.get_eval_pair()
            eval_stats.update_confmat(gt, pred)

        eval_results = eval_stats.get_eval_results()
        msg = '==> %s Results [%d images]: %s\n' % (phase, len(self.data_loader[phase]), eval_results[0])
        msg += 'Per-class IoU:\n'
        msg += ''.join(['%s: %.2f\n' % (cname,ciu)
                        for cname,ciu in zip(self.data_loader['train'].dataset.label2name, eval_results[1])])
        print(msg)
        with open(self.visualizer.log_name, "a") as log_file:
            log_file.write('%s\n' % msg)

        return eval_results[0]['Mean IoU']
