import numpy as np
import torch
import os
from torch.autograd import Variable
from .trainer import BaseTrainer
from . import networks
from util.image_pool import ImagePool


class AmortCrossEntTrainer(BaseTrainer):
    def name(self):
        return 'AmortCrossEntTrainer'

    def __init__(self, opt):
        super(AmortCrossEntTrainer, self).__init__(opt)

        self._set_model(opt)

        if self.isTrain:
            self.train(mode=True) # dropout turns on
            self._set_loss()
            self._set_optim(opt)

            print('------------ Networks initialized -------------')
            networks.print_network(self.models['G_A'])
            print('-----------------------------------------------')
            networks.print_network(self.models['D_B'])

    def _set_model(self, opt):
        self.models['G_A'] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                               opt.norm, opt.use_dropout, self.gpu_ids, 'softmax') # G_A(A)
        if self.isTrain:
            self.models['D_B'] = networks.define_D(opt.output_nc+opt.input_nc, opt.ndf, # D_B((A,B))
                                                   opt.which_model_netD,
                                                   opt.n_layers_D, opt.norm, opt.gan_type, self.gpu_ids)

    def _set_loss(self):
        self.lossfuncs['CE'] = torch.nn.NLLLoss2d(ignore_index=self.opt.ignore_index)
        if len(self.gpu_ids) > 0:
            self.lossfuncs['CE'] = self.lossfuncs['CE'].cuda(self.gpu_ids[0])

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        real_B_onehot = self.compute_real_B_onehot() # one-hot real_B, dtype=Float
        self.real_pair = torch.cat((self.real_A, real_B_onehot), dim=1)

    def backward_G_A(self):
        self.fake_B = self.models['G_A'].forward(self.real_A)

        # cross_ent(G_A(A), B)
        self.losses['G_A-CE'] = self.lossfuncs['CE'](self.fake_B, self.real_B)

        # ( CE - D_B(A, G_A(A)) )^2
        fake_pair = torch.cat((self.real_A, self.fake_B), dim=1)
        D_B_fake_pair = self.models['D_B'].forward(fake_pair)
        self.losses['G_A-diff'] = torch.mean( torch.pow(self.losses['G_A-CE'] - D_B_fake_pair, 2) )

        # lambda_A * cross_ent + diff^2
        self.losses['G_A'] = self.losses['G_A-CE'] + self.losses['G_A-diff']*self.opt.lambda_A
        self.losses['G_A'].backward()

    def backward_D_B(self):
        self.fake_B = self.models['G_A'].forward(self.real_A) # since G_A has updated

        # cross_ent(G_A(A), B)
        self.losses['D_B-CE'] = self.lossfuncs['CE'](self.fake_B, self.real_B)

        fake_pair = torch.cat((self.real_A, self.fake_B.detach()), dim=1)
        D_B_fake_pair = self.models['D_B'].forward(fake_pair)
        D_B_real_pair = self.models['D_B'].forward(self.real_pair)

        self.losses['D_B-diffF'] = torch.mean( torch.pow(self.losses['D_B-CE'] - D_B_fake_pair, 2) )
        self.losses['D_B-diffR'] = torch.mean( torch.pow(self.losses['D_B-CE'] - D_B_real_pair, 2) )

        # diff_R^2 - lambda_B * diff_F^2
        self.losses['D_B'] = self.losses['D_B-diffR'] - self.losses['D_B-diffF']*self.opt.lambda_B
        self.losses['D_B'].backward()

    def backward(self):
        # G_A
        for _ in range(3):
            self.optimizers['G_A'].zero_grad()
            self.backward_G_A()
            self.optimizers['G_A'].step()

        # D_B
        self.optimizers['D_B'].zero_grad()
        self.backward_D_B()
        self.optimizers['D_B'].step()

    def on_begin_epoch(self, epoch):
        #if epoch > 1 and epoch % 50 == 0:
        #    self.trainer.opt.lambda_B *= 1e2
        #if epoch == 1:
        #    self.opt.lambda_A = 0
        #    self.opt.lambda_B = 0
        #elif epoch == 50:
        #    self.opt.lambda_A = 1
        #    self.opt.lambda_B = 1
        #elif epoch == 800:
        #    self.opt.lambda_B = 10000
        pass

    def compute_real_B_onehot(self):
        opt = self.opt
        real_B_int = self.input_B.unsqueeze(dim=1)
        real_B_onehot = self.Tensor(opt.batchSize, opt.output_nc, opt.heightSize, opt.widthSize)
        real_B_onehot.zero_()
        real_B_onehot.scatter_(1, real_B_int, 1)

        if len(self.gpu_ids) > 0:
            real_B_onehot = real_B_onehot.cuda(self.gpu_ids[0])
        real_B_onehot = Variable(real_B_onehot)
        return real_B_onehot

    def get_current_losses(self):
        return { key:val.data[0] for key,val in self.losses.iteritems() }

    def get_current_visuals(self):
        (pred, gt) = self.get_eval_pair()
        return {'real_A': self.real_A.data.cpu().float().numpy(), 'real_B': gt, 'fake_B': pred}
