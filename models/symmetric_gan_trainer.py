import numpy as np
import torch
import os
from torch.autograd import Variable
from .trainer import BaseTrainer
from . import networks
from util.image_pool import ImagePool

TAU = 0.9

class SymmetricGANCETrainer(BaseTrainer):
    def name(self):
        return 'SymmetricGANCETrainer'

    def __init__(self, opt):
        super(SymmetricGANCETrainer, self).__init__(opt)

        self._set_model(opt)

        if self.isTrain:
            self.train(mode=True) # dropout turns out
            self._set_loss()
            self._set_optim(opt)
            self._set_fake_pool()
            print('------------ Networks initialized -------------')
            networks.print_network(self.models['G_A'])
            networks.print_network(self.models['G_B'])
            networks.print_network(self.models['D_A'])
            print('-----------------------------------------------')

    def _set_model(self, opt):
        self.models['G_A'] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                               opt.norm, opt.use_dropout, self.gpu_ids, 'softmax') # G_A(A)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # TODO: G_B outputs an intermediate representation rather than down to A, e.g., bottleneck of G_A
            self.models['G_B'] = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG,
                                                   opt.norm, opt.use_dropout, self.gpu_ids, 'tanh') # G_B(B)
            self.models['D_A'] = networks.define_D(opt.input_nc, opt.ndf, # D_A(A)
                                                   opt.which_model_netD, # TODO: use a different netD for D_B
                                                   opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

    def _set_loss(self):
        self.lossfuncs['CE'] = torch.nn.NLLLoss2d(ignore_index=self.opt.ignore_index)
        self.lossfuncs['L1'] = torch.nn.L1Loss()
        self.lossfuncs['GAN_A'] = networks.GANLoss(use_lsgan=not self.opt.no_lsgan, tensor=self.Tensor) # GAN on A
        if len(self.gpu_ids) > 0:
            self.lossfuncs['CE'] = self.lossfuncs['CE'].cuda(self.gpu_ids[0])
            self.lossfuncs['L1'] = self.lossfuncs['L1'].cuda(self.gpu_ids[0])
            self.lossfuncs['GAN_A'] = self.lossfuncs['GAN_A'].cuda(self.gpu_ids[0])

    def _set_fake_pool(self):
        self.fake_pool['A'] = ImagePool(self.opt.pool_size)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_B_onehot = self.compute_real_B_onehot() # one-hot real_B, dtype=Float

        self.fake_B = self.models['G_A'].forward(self.real_A) # G_A(A)
        self.rec_A = self.models['G_B'].forward(self.fake_B) # G_B(G_A(A))
        self.fake_A = self.models['G_B'].forward(self.real_B_onehot) # G_B(B)
        self.rec_B = self.models['G_A'].forward(self.fake_A) # G_A(G_B(B))

    def backward_D_basic(self, netD, gan_loss, real, fake):
        ''' gan_loss: instance of GANLoss
        '''
        # Real: log( D(real) )
        pred_real = netD.forward(real) # each element should close to 1 to be real
        loss_D_real = gan_loss(pred_real, True)

        # Fake: log( 1 - D(fake) )
        pred_fake = netD.forward(fake.detach()) # detach() so back-prop not affect G
        loss_D_fake = gan_loss(pred_fake, False)

        # Combined loss: max_D 1/2 [log D(real) + log (1 - D(fake))]
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_B(self):
        # CE(realB, fakeB)
        self.losses['G_A-CE'] = self.lossfuncs['CE'](self.fake_B, self.real_B)

        # CE(realB, recB)
        self.losses['G_B-G_A-CE'] = self.lossfuncs['CE'](self.rec_B, self.real_B)

    def backward_A(self):
        # L1( G_B(real_B), A )
        self.losses['G_A-L1'] = self.lossfuncs['L1'](self.fake_A, self.real_A)

        # L1( G_B(G_A(real_A)), A )
        self.losses['G_A-G_B-L1'] = self.lossfuncs['L1'](self.rec_A, self.real_A)

        # GAN( G_B(G_A(A)), fake )
        pred_rec = self.models['D_A'].forward(self.rec_A)
        self.losses['G_A-G_B-GAN'] = self.lossfuncs['GAN_A'](pred_rec, True) * 0.5
        pred_fake = self.models['D_A'].forward(self.fake_A)
        self.losses['G_A-G_B-GAN'] += self.lossfuncs['GAN_A'](pred_fake, True) * 0.5

    def backward_D_A(self):
        # rec_A is self.rec_A with random replacements from fake_A_pool
        rec_A = self.fake_pool['A'].query(self.rec_A)
        fake_A = self.fake_pool['A'].query(self.fake_A)

        # D_A(A)
        self.losses['D_A'] = self.backward_D_basic(self.models['D_A'], self.lossfuncs['GAN_A'],
                                                   self.real_A, rec_A)
        self.losses['D_A'] += self.backward_D_basic(self.models['D_A'], self.lossfuncs['GAN_A'],
                                                   self.real_A, fake_A)
        self.losses['D_A'].backward()

    def backward_G_AB(self):
        self.backward_A()
        self.backward_B()

        self.losses['G_AB'] = (self.losses['G_A-L1'] + self.losses['G_A-G_B-L1'] + self.losses['G_A-G_B-GAN']) * self.opt.lambda_A

        if self.use_real_B:
            self.losses['G_AB'] += (self.losses['G_B-G_A-CE'] + self.losses['G_A-CE']) * self.opt.lambda_B

        self.losses['G_AB'].backward()

    def backward(self):
        # G_A and G_B
        self.optimizers['G_A'].zero_grad()
        self.optimizers['G_B'].zero_grad()
        self.backward_G_AB()
        self.optimizers['G_B'].step()
        self.optimizers['G_A'].step()

        # D_A
        self.optimizers['D_A'].zero_grad()
        self.backward_D_A()
        self.optimizers['D_A'].step()

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
        return {'real_A': self.real_A.data.cpu().float().numpy(),
                'rec_A': self.rec_A.data.cpu().float().numpy(),
                'fake_A': self.fake_A.data.cpu().float().numpy(),
                'rec_B': self.rec_B.data.cpu().numpy().argmax(1),
                'real_B': gt,
                'fake_B': pred}
