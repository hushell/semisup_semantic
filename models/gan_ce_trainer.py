import numpy as np
import torch
import os
from torch.autograd import Variable
from .trainer import BaseTrainer
from . import networks
from util.image_pool import ImagePool

TAU = 0.9

class GANCrossEntTrainer(BaseTrainer):
    def name(self):
        return 'GANCrossEntTrainer'

    def __init__(self, opt):
        super(GANCrossEntTrainer, self).__init__(opt)

        self._set_model(opt)

        if self.isTrain:
            self.train(mode=True) # dropout turns out
            self._set_loss()
            self._set_optim(opt)
            self._set_fake_pool()
            print('------------ Networks initialized -------------')
            networks.print_network(self.models['G_A'])
            print('-----------------------------------------------')
            networks.print_network(self.models['D_B'])

    def _set_model(self, opt):
        self.models['G_A'] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                               opt.norm, opt.use_dropout, self.gpu_ids, 'softmax') # G_A(A)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.models['D_B'] = networks.define_D(opt.output_nc, opt.ndf, # D_B(B)
                                                   opt.which_model_netD,
                                                   opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

    def _set_loss(self):
        self.lossfuncs['CE'] = torch.nn.NLLLoss2d()
        self.lossfuncs['GAN_B'] = networks.GANLoss(use_lsgan=not self.opt.no_lsgan, tensor=self.Tensor) # GAN on B
        if len(self.gpu_ids) > 0:
            self.lossfuncs['CE'] = self.lossfuncs['CE'].cuda(self.gpu_ids[0])
            self.lossfuncs['GAN_B'] = self.lossfuncs['GAN_B'].cuda(self.gpu_ids[0])

    def _set_fake_pool(self):
        self.fake_pool['B'] = ImagePool(self.opt.pool_size)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        #self.fake_B = self.models['G_A'].forward(self.real_A)

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

    def backward_D_B(self):
        # self.fake_B = G_A(A), fake_B is self.fake_B with random replacements from fake_B_pool
        #fake_B = self.fake_pool['B'].query(self.fake_B)
        self.fake_B = self.models['G_A'].forward(self.real_A) # since G_A has updated

        # one-hot real_B (or + noise), dtype=Float
        self.compute_real_B_onehot(self.fake_B)

        # D_B(B)
        self.losses['D_B'] = self.backward_D_basic(self.models['D_B'], self.lossfuncs['GAN_B'],
                                                   self.real_B_onehot, self.fake_B)
        self.losses['D_B'].backward()

    def backward_G_A(self):
        self.fake_B = self.models['G_A'].forward(self.real_A)

        # cross_ent(G_A(A), B)
        self.losses['G_A-CE'] = self.lossfuncs['CE'](self.fake_B, self.real_B)

        # suppose: min_G 1/2 log(1 - D(fake))
        # non-saturating: max_G 1/2 log(D(fake)), fake = G_A(A)
        pred_fake = self.models['D_B'].forward(self.fake_B) # back-prop will flow fake_B = G_A(A)
        self.losses['G_A-GAN'] = self.lossfuncs['GAN_B'](pred_fake, True) * 0.5

        self.losses['G_A'] = self.losses['G_A-CE']*self.opt.lambda_A + self.losses['G_A-GAN']
        self.losses['G_A'].backward()

    def backward(self):
        # G_A
        self.optimizers['G_A'].zero_grad()
        self.backward_G_A()
        self.optimizers['G_A'].step()

        # D_B
        self.optimizers['D_B'].zero_grad()
        self.backward_D_B()
        self.optimizers['D_B'].step()

    def compute_real_B_onehot(self, fake_B):
        if not self.opt.gt_noise:
            #real_B_onehot = np.eye(opt.output_nc)[real_B_int]
            real_B_int = self.input_B.unsqueeze(dim=1)
            real_B_onehot = self.Tensor(fake_B.data.size())
            real_B_onehot.zero_()
            real_B_onehot.scatter_(1, real_B_int, 1)
        else:
            real_B_int = self.input_B.cpu().numpy()
            fake_B_cpy = fake_B.data.cpu().numpy()
            nn,hh,ww = np.meshgrid(np.arange(fake_B_cpy.shape[0]), np.arange(fake_B_cpy.shape[2]),
                                   np.arange(fake_B_cpy.shape[3]), indexing='ij')
            S = fake_B_cpy[nn,real_B_int,hh,ww] # fake_B(real_B)
            Y = np.maximum(TAU, S) # threshold by 0.9

            #coeff = (1-Y) / (1-S) # renormalize to 1
            one_minus_S = 1.0-S
            msk = np.isclose(one_minus_S, 0.0)
            one_minus_S[msk] = 1.0
            coeff = (1-Y) / one_minus_S
            assert(not np.any(np.isnan(coeff)) and not np.any(np.isinf(coeff)))
            coeff = coeff[:,np.newaxis,...]

            real_B_onehot = fake_B_cpy * coeff
            real_B_onehot[nn,real_B_int,hh,ww] = Y
            real_B_onehot = torch.from_numpy(real_B_onehot)
        if len(self.gpu_ids) > 0:
            real_B_onehot = real_B_onehot.cuda(self.gpu_ids[0])
        self.real_B_onehot = Variable(real_B_onehot)

    def get_current_losses(self):
        return { key:val.data[0] for key,val in self.losses.iteritems() }

    def get_current_visuals(self):
        (pred, gt) = self.get_eval_pair()
        return {'real_A': self.real_A.data.cpu().float().numpy(), 'real_B': gt, 'fake_B': pred}
