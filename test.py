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
if len(opt.gpu_ids) > 0:
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

opt.gpu_ids = range(0,len(opt.gpu_ids)) # new range starting from 0
opt.batchSize = 1

######################################
# data_loaders, visualizer, trainer(models, optimizers)
opt.phase = 'val'
opt.isTrain = False
val_loader = CreateDataLoader(opt)
opt = val_loader.update_opt(opt)

opt.isTrain = True # to load all nets
visualizer = Visualizer(opt)
trainer = CreateTrainer(opt)

######################################
# exp_manager
expmgr = ExperimentManager(opt, trainer, visualizer, train_loader=val_loader, test_loader=val_loader) # need a dummy train_loader
expmgr.resume(opt.which_epoch)

######################################
# evaluation
val_acc = expmgr.evaluation('test')

