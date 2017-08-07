import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import imp
import torchnet as tnt


LUT = [(40,0.0001), (100,0.00003), (160,0.00001), (220,0.000003), (240,0.000001)]

class GANCrossEntModel(BaseModel):
    def name(self):
        return 'GANCrossEntModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        self.input_A = self.Tensor(nb, opt.input_nc, opt.heightSize, opt.widthSize)
        self.input_B = self.Tensor(nb, opt.heightSize, opt.widthSize)

        # load/define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.lr_scheme = opt.lr_scheme
            self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionCE = torch.nn.NLLLoss2d()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if len(self.gpu_ids) > 0:
                self.criterionCE = self.criterionCE.cuda()
                self.criterionGAN = self.criterionGAN.cuda()

            # initialize optimizers
            parameters = [p for p in self.netG_A.parameters() if p.requires_grad]
            if opt.optim_method == 'adam':
                self.optimizer_G = torch.optim.Adam(itertools.chain(parameters), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim_method == 'sgd':
                self.optimizer_G = torch.optim.SGD(itertools.chain(parameters), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
            else:
                raise ValueError("Optim_method [%s] not recognized." % opt.optim_method)
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            networks.print_network(self.netD_A)
            print('-----------------------------------------------')

        # visualization
        cslabels = imp.load_source("",'./datasets/cityscapes_AB/labels.py')
        label2trainId = np.asarray([(1+label.trainId) if label.trainId < 255 else 0 for label in cslabels.labels], dtype=np.float32)
        label2color = np.asarray([(label.color) for label in cslabels.labels], dtype=np.uint8)
        num_cats      = 1+19 # the first extra category is for the pixels with missing category
        trainId2labelId = np.ndarray([num_cats], dtype=np.int32)
        trainId2labelId.fill(-1)
        for labelId in range(len(cslabels.labels)):
            trainId = int(label2trainId[labelId])
            if trainId2labelId[trainId] == -1:
                trainId2labelId[trainId] = labelId
        self.trainId2color = label2color[trainId2labelId]
        clsNames = np.asarray([label.name for label in cslabels.labels], dtype=np.str)
        self.trainId2name = clsNames[trainId2labelId]

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']

        if len(self.gpu_ids) > 0:
            self.input_A = input_A.cuda()
            self.input_B = input_B.long().cuda()
        else:
            self.input_A = input_A
            self.input_B = input_B.long()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real: log( D(real) )
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake: log( 1 - D(fake) )
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss: max_D 1/2 [log D(real) + log (1 - D(fake))]
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        # self.fake_B = G_A(A), fake_B is self.fake_B with random replacements from fake_B_pool
        fake_B = self.fake_B_pool.query(self.fake_B)

        real_B_int = self.input_B.unsqueeze(dim=1)
        real_B_onehot = self.Tensor(fake_B.data.size())
        real_B_onehot.zero_()
        real_B_onehot.scatter_(1, real_B_int, 1)
        real_B_onehot = Variable(real_B_onehot)

        self.loss_D_A = self.backward_D_basic(self.netD_A, real_B_onehot, fake_B)

    def backward_G(self):
        self.fake_B = self.netG_A.forward(self.real_A)
        # cross_ent(G_A(A), B)
        self.loss_G_A_CE = self.criterionCE(self.fake_B, self.real_B)
        # suppose: min_G 1/2 log(1 - D(fake))
        # non-saturating: max_G 1/2 log(D(fake))
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A_GAN = self.criterionGAN(pred_fake, True) * 0.5
        self.loss_G_A = self.loss_G_A_CE * self.opt.lambda_A + self.loss_G_A_GAN
        self.loss_G_A.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

    def get_current_errors(self):
        Total = self.loss_G_A.data[0]
        CE = self.loss_G_A_CE.data[0]
        GAN = self.loss_G_A_GAN.data[0]
        return OrderedDict([('G_Total', Total), ('G_CE', CE), ('G_GAN', GAN)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2lab(self.fake_B.data, self.trainId2color)
        real_B = util.tensor2lab(self.real_B.data, self.trainId2color)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def get_pred_gt(self):
        logits      = self.fake_B.data.cpu().numpy()
        predictions = logits.argmax(1).squeeze()
        groundtruth = self.real_B.data.cpu().numpy()
        return predictions, groundtruth

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        if self.lr_scheme == 'linear':
            if epoch > self.opt.niter:
                lrd = self.opt.lr / self.opt.niter_decay
                lr = self.old_lr - lrd
            else:
                lr = self.opt.lr
        elif self.lr_scheme == 'lut':
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
        else:
            raise ValueError("lr scheme [%s] not recognized." % opt.lr_scheme)

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr

        print('===> Update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
