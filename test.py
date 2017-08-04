import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import torch
from util.meter import SegmentationMeter

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# to prevent opencv from initializing CUDA in workers
torch.randn(8).cuda()
os.environ['CUDA_VISIBLE_DEVICES'] = ''

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
vidx = 0
eval_stats = SegmentationMeter(n_class=opt.output_nc)
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()

    # visuals
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)
    #visualizer.display_current_results(visuals, opt.which_epoch, vidx) # localhost:port
    vidx += len(visuals.keys())

    # eval results
    pred,gt = model.get_pred_gt()
    eval_stats.update_confmat(gt, pred)

test_results = eval_stats.get_eval_results()
msg = '==> %s Results [%d images]: %s\n' % (opt.phase, len(dataset), test_results[0])
msg += ''.join(['%s: %.2f,' % (cname,ciu) for cname,ciu in zip(model.trainId2name,test_results[1])])
print(msg)
with open(visualizer.log_name, "a") as log_file:
    log_file.write('%s\n' % msg)
webpage.save()
