import numpy as np
import os
import ntpath
import time
from . import util
from . import html

class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.port)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('===> Visualizer.__init__(): create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, it, do_save=False, idx=1):
        if self.display_id > 0: # show images in the browser
            for label, image_numpy in visuals.items():
                #image_numpy = np.flipud(image_numpy)
                self.vis.image(image_numpy, opts=dict(title='%d,%d: %s' % (epoch,it,label)),
                               win=self.display_id + idx)
                idx += 1

        # save images to a html file
        if self.use_html and do_save:
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # plot metrics
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

    # print metrics
    def print_current_metrics(self, epoch, total_i, t, metrics):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, total_i, t)
        for k, v in metrics.items():
            dformat = '%.e' if v < 1e-3 else '%.3f'
            message += '%s: {} '.format(dformat) % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
