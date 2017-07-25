import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.meter import *
import os
import torch

opt = TrainOptions().parse()
opt.nThreads = max(opt.batchSize / 2, 2)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# to prevent opencv from initializing CUDA in workers
torch.randn(8).cuda()
os.environ['CUDA_VISIBLE_DEVICES'] = ''

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
dataset_size = dataset_size - (dataset_size % opt.batchSize)
print('#training images = %d' % len(data_loader))

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0 if not opt.continue_train else int(opt.which_epoch)*dataset_size

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    model.update_learning_rate(epoch)

    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            # compute losses
            errors = model.get_current_errors()
            # compute eval result for current batch
            eval_stats = DAverageMeter()
            confMeter = model.get_eval_results()
            eval_stats.update({'confMeter': confMeter})
            # plot errors, metrics
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
                visualizer.plot_current_metrics(epoch, float(epoch_iter)/dataset_size, opt, eval_stats.average()['confMeter'])
            # print errors & metrics
            errors = dict(errors, **eval_stats.average()['confMeter'])
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
