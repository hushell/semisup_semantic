import numpy as np
import torch
import os
from torch.autograd import Variable
from .trainer import BaseTrainer
from . import networks
from util.image_pool import ImagePool

G_A_init_weight_path = './checkpoints/dropout_camvid_cross_ent_st_resnet_9blocks_netD4_b4/G_A_net_1000.pth'
N_LOSS_AUG_ITER = 3
N_INFER_ITER = 30
LR_INFER = 1e-1
ON_DEBUG_MODE = True

class AmortStructSVMTrainer(BaseTrainer):
    def name(self):
        return 'AmortStructSVMTrainer'

    def __init__(self, opt):
        super(AmortStructSVMTrainer, self).__init__(opt)

        self._set_model(opt)

        if self.isTrain:
            self.train(mode=True) # dropout turns on
            self._set_loss()

            lr_coeffs = {k:0.1 if 'G' in k else 10.0 for k in self.models.keys()}
            self._set_optim(opt, lr_coeffs=lr_coeffs)

            print('------------ Networks initialized -------------')
            networks.print_network(self.models['G_A'])
            print('-----------------------------------------------')
            networks.print_network(self.models['D_B'])

    def _set_model(self, opt):
        self.models['G_test'] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                                  opt.norm, opt.use_dropout, self.gpu_ids, 'softmax')
        if os.path.exists(G_A_init_weight_path):
            state = torch.load(G_A_init_weight_path, map_location=lambda storage, loc: storage)
            self.models['G_test'].load_state_dict(state)
            print('==> Load G_test from %s' % (G_A_init_weight_path))

        if self.isTrain:
            self.models['G_A'] = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                                   opt.norm, opt.use_dropout, self.gpu_ids, 'softmax') # G_A(A)
            if os.path.exists(G_A_init_weight_path):
                state = torch.load(G_A_init_weight_path, map_location=lambda storage, loc: storage)
                self.models['G_A'].load_state_dict(state)
                print('==> Load G_A from %s' % (G_A_init_weight_path))

            self.models['D_B'] = networks.define_D(opt.output_nc+opt.input_nc, opt.ndf, # D_B( (A,B) )
                                                   opt.which_model_netD,
                                                   opt.n_layers_D, opt.norm, opt.gan_type, self.gpu_ids)

    def _set_loss(self):
        self.lossfuncs['CE'] = torch.nn.NLLLoss2d()
        if len(self.gpu_ids) > 0:
            self.lossfuncs['CE'] = self.lossfuncs['CE'].cuda(self.gpu_ids[0])

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        real_B_onehot = self.compute_real_B_onehot() # one-hot real_B, dtype=Float
        self.real_pair = torch.cat((self.real_A, real_B_onehot), dim=1)

    def test_Gtest(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        msg = 'test %s: (f,CE) =' % os.path.basename(self.image_paths[0])
        for i in range(N_INFER_ITER): # TODO: more iters?
            self.fake_B = self.models['G_test'].forward(self.real_A)

            fake_pair = torch.cat((self.real_A, self.fake_B), dim=1)
            D_B_fake_pair = self.models['D_B'].forward(fake_pair)
            loss_test = torch.mean( -D_B_fake_pair )

            self.optimizers['G_test'].zero_grad()
            loss_test.backward()
            self.optimizers['G_test'].step()

            if ON_DEBUG_MODE:
                CE_temp = self.lossfuncs['CE'](self.fake_B, self.real_B)
                msg += ' (%f,%f),' % (-loss_test.data[0], CE_temp.data[0]) # check if f and CE are correlated
            else:
                msg += ' (%f),' % (-loss_test.data[0])
        print(msg)

    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B, volatile=True)

        y_0 = self.models['G_test'].forward(self.real_A)
        self.fake_B = Variable(y_0.data.clone(), requires_grad=True)
        self.fake_B.grad = Variable(y_0.data.new().resize_as_(y_0.data).zero_())

        msg = 'test %s: f =' % os.path.basename(self.image_paths[0])
        for i in range(N_INFER_ITER):
            fake_pair = torch.cat((self.real_A, self.fake_B), dim=1)
            D_B_fake_pair = self.models['D_B'].forward(fake_pair)
            D_B_val = torch.mean( D_B_fake_pair )

            self.fake_B.grad.data.zero_()
            D_B_val.backward()
            self.fake_B.data.add_(LR_INFER, self.fake_B.grad.data)

            if ON_DEBUG_MODE:
                msg += ' %f,' % (D_B_val.data[0])
        print(msg)

    def backward_G_A(self):
        ''' loss-augmented inference by G_A
        '''
        # G_A(A)
        self.fake_B = self.models['G_A'].forward(self.real_A)

        # cross_ent(G_A(A), B)
        self.losses['G_A-CE'] = self.lossfuncs['CE'](self.fake_B, self.real_B)

        # D_B( (A, G_A(A)) )
        fake_pair = torch.cat((self.real_A, self.fake_B), dim=1)
        D_B_fake_pair = self.models['D_B'].forward(fake_pair)

        # LL_G := - D_B( (A, G_A(A)) ) - lambda_B * CE
        self.losses['LL_G'] = torch.mean( -D_B_fake_pair - self.opt.lambda_B * self.losses['G_A-CE'] )
        self.losses['LL_G'].backward()

        if False:
            f_val = torch.mean(D_B_fake_pair)
            print('backward_G_A(): G_A-CE = %f, -f = %f, LL_G = %f' % (self.losses['G_A-CE'].data[0], -f_val.data[0], self.losses['LL_G'].data[0]))

    def backward_D_B(self):
        # current most-violated
        self.fake_B = self.models['G_A'].forward(self.real_A) # since G_A has updated

        # cross_ent(G_A(A), B)
        self.losses['G_A-CE'] = self.lossfuncs['CE'](self.fake_B, self.real_B)

        # D_B( (A, G_A(A)) ), D_B( (A, B) )
        fake_pair = torch.cat((self.real_A, self.fake_B.detach()), dim=1)
        D_B_fake_pair = self.models['D_B'].forward(fake_pair)
        D_B_real_pair = self.models['D_B'].forward(self.real_pair)

        # LL_f := [f_G - lambda_B * CE - f_R ]_+
        self.losses['LL_f'] = torch.clamp(D_B_fake_pair - D_B_real_pair + self.opt.lambda_B * self.losses['G_A-CE'], min=0)
        self.losses['LL_f'] = torch.mean( self.losses['LL_f'] )
        self.losses['LL_f'].backward()

        if ON_DEBUG_MODE:
            self.losses['f_diff'] = torch.mean( D_B_fake_pair - D_B_real_pair )

    def backward(self):
        # G_A
        for _ in range(N_LOSS_AUG_ITER):
            self.optimizers['G_A'].zero_grad()
            self.backward_G_A()
            self.optimizers['G_A'].step()

        # D_B
        self.optimizers['D_B'].zero_grad()
        self.backward_D_B()
        self.optimizers['D_B'].step()

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