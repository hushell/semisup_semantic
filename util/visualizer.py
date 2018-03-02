import numpy as np
import os
import sys
import ntpath
import time
from . import util
from . import html
from collections import OrderedDict
import torchvision.utils as vutils

class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.port)

        if self.use_html:
            self.res_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results')
            train_dir = os.path.join(self.res_dir, 'train')
            val_dir = os.path.join(self.res_dir, 'val')
            test_dir = os.path.join(self.res_dir, 'test')
            print('===> Create results directory: %s' % self.res_dir)
            util.mkdirs([self.res_dir, train_dir, val_dir, test_dir])

            self.img_dict = {}

        util.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ New run at (%s) ================\n' % now)
            log_file.write('%s\n' % ' '.join(sys.argv))
            log_file.write('=================================================\n')
        print('==> OPTIONS:')
        print('%s\n' % ' '.join(sys.argv))

    def display_current_results(self, visuals, epoch, it, subset='train', do_save=0, idx=1):
        """ |visuals|: dictionary of images to display or save, do_save>1 save to webpage """
        # visdom
        if self.display_id > 0:
            for label, image_numpy in visuals.items():
                #image_numpy = np.flipud(image_numpy)
                self.vis.image(image_numpy, opts=dict(title='%s: epoch%d, iter%d: %s' % (subset,epoch,it,label)),
                               win=self.display_id + idx)
                idx += 1

        # save images
        if self.use_html and do_save > 0:
            key = '%d_%d' % (epoch, it)
            for label, image_numpy in visuals.items():
                image_name = '%s_%s.png' % (key, label)
                img_path = os.path.join(self.res_dir, subset, image_name)
                util.save_image(image_numpy.transpose((1,2,0)), img_path)

                if do_save > 1:
                    self.img_dict.setdefault(key, {})
                    self.img_dict[key][label] = img_path

    def save_webpage(self, prefix='train'):
        if not self.use_html:
            return
        webpage = html.HTML(self.res_dir, prefix, 'Experiment name = %s' % self.name, reflesh=1)
        img_dict = OrderedDict(sorted(self.img_dict.items(), key=lambda t: int(t[0].split('_')[0])))
        for key, visuals in img_dict.items():
            webpage.add_header('index [%s]' % key)
            ims = []
            txts = []
            links = []

            for label, image_numpy in visuals.items():
                img_path = '%s_%s.png' % (key, label)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            webpage.add_images(ims, txts, links, width=self.win_size)
        webpage.save()

    def plot_current_metrics(self, metrics, total_i):
        if self.display_id <= 0:
            return
        if metrics: # not empty
            if not hasattr(self, 'plot_metrics'):
                self.plot_metrics = {'Y':[], 'X':[], 'legends':metrics.keys()};
            self.plot_metrics['Y'].append(metrics.values())
            self.plot_metrics['X'].append(total_i)

        # assert self.plot_metrics exists
        if len(self.plot_metrics['Y']) > 1:
            self.vis.line(
                X=np.array(self.plot_metrics['X']),
                Y=np.array(self.plot_metrics['Y']),
                opts={
                    'title': self.name + ' metrics over iteration',
                    'legend': self.plot_metrics['legends'],
                    'xlabel': 'iteration',
                    'ylabel': 'metrics'},
                win=self.display_id)

    def print_current_metrics(self, epoch, total_i, t, metrics):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, total_i, t)
        for k, v in metrics.items():
            dformat = '%.e' if abs(v) < 1e-3 else '%.3f'
            message += '%s: {} '.format(dformat) % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

