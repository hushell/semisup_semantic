import numpy as np
import torch
import os
from torch.autograd import Variable
from .trainer import BaseTrainer
from . import networks


class CrossEntropyTrainer(BaseTrainer):
    def name(self):
        return 'CrossEntropyTrainer'

    def __init__(self, opt):
        super(CrossEntropyTrainer, self).__init__(opt)

        self._set_model(opt)

        if self.isTrain:
            self.train(mode=True) # dropout turns out
            self._set_loss()
            self._set_optim(opt)
            print('------------ Networks initialized -------------')
            networks.print_network(self.models['G_A'])
            print('-----------------------------------------------')

    def _set_model(self, opt):
        self.models['G_A'] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                               opt.norm, opt.use_dropout, self.gpu_ids, 'softmax') # G_A(A)

    def _set_loss(self):
        self.lossfuncs['CE'] = torch.nn.NLLLoss2d()
        if len(self.gpu_ids) > 0:
            self.lossfuncs['CE'] = self.lossfuncs['CE'].cuda(self.gpu_ids[0])

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.fake_B = self.models['G_A'].forward(self.real_A)

    def backward(self):
        self.optimizers['G_A'].zero_grad()
        self.losses['G_A'] = self.lossfuncs['CE'](self.fake_B, self.real_B)
        self.losses['G_A'].backward()
        self.optimizers['G_A'].step()

    def get_current_losses(self):
        return {'G_A': self.losses['G_A'].data[0]}

    def get_current_visuals(self):
        (pred, gt) = self.get_eval_pair()
        return {'real_A': self.real_A.data.cpu().float().numpy(), 'real_B': gt, 'fake_B': pred}
