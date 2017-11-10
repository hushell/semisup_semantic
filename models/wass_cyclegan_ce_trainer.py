import numpy as np
import torch
import os
from torch.autograd import Variable
from torch.autograd import grad
from .trainer import BaseTrainer
from . import networks
from util.image_pool import ImagePool

TAU = 0.9
N_D_STEPS = 5 # TODO: 100 at beginning and every 500 iters
CLAMP_LOW = -0.01
CLAMP_UPP = 0.01
LAMBDA = 10 # Gradient penalty lambda hyperparameter

class WassCycleGANCrossEntTrainer(BaseTrainer):
    def name(self):
        return 'WassCycleGANCrossEntTrainer'

    def __init__(self, opt):
        super(WassCycleGANCrossEntTrainer, self).__init__(opt)

        self._set_model(opt)

        if self.isTrain:
            self.train(mode=True) # dropout turns out
            self._set_loss()
            self._set_optim(opt)
            self._set_fake_pool()
            self.losses['G_A-CE'] = Variable(torch.from_numpy(np.array([0])))

    def _set_model(self, opt):
        self.models['G_A'] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                               opt.norm, opt.use_dropout, self.gpu_ids, 'softmax') # G_A(A)
        if self.isTrain:
            # TODO: G_B outputs an intermediate representation rather than down to A, e.g., bottleneck of G_A
            self.models['G_B'] = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG,
                                                   opt.norm, opt.use_dropout, self.gpu_ids, 'tanh') # G_B(B)
            self.models['D_A'] = networks.define_D(opt.input_nc, opt.ndf, # D_A(A)
                                                   opt.which_model_netD, # TODO: use a different netD for D_B
                                                   opt.n_layers_D, opt.norm, 'wass', self.gpu_ids)

    def _set_loss(self):
        self.lossfuncs['CE'] = torch.nn.NLLLoss2d(ignore_index=self.opt.ignore_index)
        self.lossfuncs['L1'] = torch.nn.L1Loss()

        if len(self.gpu_ids) > 0:
            self.lossfuncs['CE'] = self.lossfuncs['CE'].cuda(self.gpu_ids[0])
            self.lossfuncs['L1'] = self.lossfuncs['L1'].cuda(self.gpu_ids[0])

    def _set_fake_pool(self):
        #self.fake_pool['B'] = ImagePool(self.opt.pool_size)
        self.fake_pool['A'] = ImagePool(self.opt.pool_size)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.fake_B = self.models['G_A'].forward(self.real_A) # G_A(A)
        self.rec_A = self.models['G_B'].forward(self.fake_B) # G_B(G_A(A))

    def backward_D_A(self, rec_A, eval_mode=False):
        '''
        -mean_real(D(x)) + mean_fake(D(x))
        '''
        # train with real
        D_A_real = self.models['D_A'].forward(self.real_A)
        D_A_real = -D_A_real.mean()
        if not eval_mode:
            D_A_real.backward()

        # train with fake
        detached_rec_A = rec_A.detach()
        D_A_fake = self.models['D_A'].forward(detached_rec_A)
        D_A_fake = D_A_fake.mean()
        if not eval_mode:
            D_A_fake.backward()

        # train with gradient penalty
        if self.opt.gan_type == 'wassgp':
            gradient_penalty = self.calc_gradient_penalty(self.real_A.data, detached_rec_A.data)
            gradient_penalty.backward()

        self.losses['D_A'] = D_A_real + D_A_fake
        if self.opt.gan_type == 'wassgp':
            self.losses['D_A'] = self.losses['D_A'] + gradient_penalty
        #self.losses['D_A'].backward() # NOTE: we can do backward() here, but will use more MEM

    def backward_G_AB(self, eval_mode=False):
        '''
        lbdaA * L1 + lbdaB * CE - mean_fake(D_A(rec))
        '''
        # TODO: another update for G_B with fake_A

        # -mean_fake( D(G_B(G_A(A))) )
        minus_D_A_fake = self.models['D_A'].forward(self.rec_A)
        minus_D_A_fake = -minus_D_A_fake.mean()
        self.losses['G_A-G_B-GAN'] = minus_D_A_fake

        # L1( G_B(G_A(A)), A )
        self.losses['G_A-G_B-L1'] = self.lossfuncs['L1'](self.rec_A, self.real_A)

        # reconstruction loss for A
        self.losses['G_AB'] = self.losses['G_A-G_B-L1']*self.opt.lambda_A + self.losses['G_A-G_B-GAN']

        if self.use_real_B:
            # cross_ent(G_A(A), B)
            self.losses['G_A-CE'] = self.lossfuncs['CE'](self.fake_B, self.real_B)
            self.losses['G_AB'] += self.losses['G_A-CE']*self.opt.lambda_B

        if not eval_mode:
            self.losses['G_AB'].backward()

    def backward(self):
        # D_A
        for di in range(N_D_STEPS):
            enqueue_rec_A = self.rec_A if di == 0 else None
            rec_A = self.fake_pool['A'].query(enqueue_rec_A)
            self.optimizers['D_A'].zero_grad()
            self.backward_D_A(rec_A)
            self.optimizers['D_A'].step()

            # clamp parameters to a cube
            if self.opt.gan_type is not 'wassgp':
                for p in self.models['D_A'].parameters():
                    p.data.clamp_(CLAMP_LOW, CLAMP_UPP)

        # G_A and G_B
        self.optimizers['G_A'].zero_grad()
        self.optimizers['G_B'].zero_grad()
        self.backward_G_AB()
        self.optimizers['G_B'].step()
        self.optimizers['G_A'].step()

    def calc_gradient_penalty(self, real_data, fake_data):
        # print "real_data: ", real_data.size(), fake_data.size()
        batchSize = self.opt.batchSize
        use_cuda = len(self.gpu_ids) > 0

        alpha = torch.rand(batchSize, 1)
        alpha = alpha.expand(batchSize, real_data.nelement()/batchSize).contiguous().view(self.real_A.size())
        alpha = alpha.cuda(self.gpu_ids[0]) if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if use_cuda > 0:
            interpolates = interpolates.cuda(self.gpu_ids[0])
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.models['D_A'](interpolates)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu_ids[0]) if use_cuda
                             else torch.ones(disc_interpolates.size()),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

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
        res = {'real_A': self.real_A.data.cpu(),
                'real_B': gt, 'fake_B': pred}
        if hasattr(self, 'rec_A'):
            res['rec_A'] = self.rec_A.data.cpu()
        return res

    def test(self, phase='train'):
        super(WassCycleGANCrossEntTrainer, self).test()
        if phase == 'test': # in training's eval, don't need rec_A
            self.rec_A = self.models['G_B'].forward(self.fake_B) # G_B(G_A(A))
