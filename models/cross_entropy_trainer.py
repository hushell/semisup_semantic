import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from util.image_pool import ImagePool
from .trainer import BaseTrainer
from . import networks
import torchnet as tnt


class CrossEntropyTrainer(BaseTrainer):
    def name(self):
        return 'CrossEntropyTrainer'

    def __init__(self, opt):
        super(CrossEntropyTrainer, self).__init__(opt)

        self._set_model(opt)

        if self.isTrain:
            self._set_loss()
            self._set_optim(opt)
            print('------------ Networks initialized -------------')
            networks.print_network(self.models['G_A'])
            print('-----------------------------------------------')

    def _set_model(self, opt):
        # load/define networks, Code (paper): G_A (G)
        self.models['G_A'] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                                  opt.norm, opt.use_dropout, self.gpu_ids)

    def _set_loss(self):
        self.lossfuncs['G_CE'] = torch.nn.NLLLoss2d()
        if len(self.gpu_ids) > 0:
            self.lossfuncs['G_CE'] = self.lossfuncs['G_CE'].cuda(self.gpu_ids[0])

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.fake_B = self.models['G_A'].forward(self.real_A)

    def backward(self):
        self.losses['G_A'] = self.lossfuncs['G_CE'](self.fake_B, self.real_B)
        self.losses['G_A'].backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # backward
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_losses(self):
        return {'G_A': self.losses['G_A'].data[0]}

    def get_current_visuals(self):
        (pred, gt) = self.get_eval_pair()
        return {'real_A': self.real_A.data.cpu().float().numpy(), 'real_B': gt, 'fake_B': pred}
