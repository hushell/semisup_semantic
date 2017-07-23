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


class GeneratorOnlyModel(BaseModel):
    def name(self):
        return 'GeneratorOnlyModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # Code (paper): G_A (G)
        assert(opt.which_model_netG == 'resnet_softmax_9blocks')
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterion = torch.nn.NLLLoss2d()
            #self.criterion = torch.nn.L1Loss()
            if len(self.gpu_ids) > 0:
                self.criterion = self.criterion.cuda()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
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

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        #import ipdb; ipdb.set_trace()
        #self.input_A.resize_(input_A.size()).copy_(input_A)
        #self.input_B.resize_(input_B.size()).copy_(input_B)
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
        self.fake_B = self.netG_A.forward(self.real_A)

    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        # loss G_A(A)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.loss_G_A = self.criterion(self.fake_B, self.real_B)
        self.loss_G_A.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        G_A = self.loss_G_A.data[0]
        return OrderedDict([('G_A', G_A), ('Nothing', 0)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2lab(self.fake_B.data, self.trainId2color)
        real_B = util.tensor2lab(self.real_B.data, self.trainId2color)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def get_eval_results(self):
        predictions = self.fake_B.data.transpose(1, len(self.fake_B.data.size())-1) # NCHW -> NHWC
        predictions = predictions.contiguous().view(-1, predictions.size(-1)) # NHWC -> (N*H*W)C
        groundtruth = self.real_B.data.contiguous().view(-1)

        # -- hacks here: make sure that you do not consider the first
        # category which is for pixels with missing annotation.
        num_cats = predictions.size(1) - 1 # the first category is for pixels with missing annotation
        groundtruth.add_(-1)

        import ipdb; ipdb.set_trace()
        # The first category (label -1) is for pixels with missing annotation
        valid = torch.ge(groundtruth, 0)
        groundtruth = torch.masked_select(groundtruth, valid)
        preds_tuple = predictions.split(split_size=1, dim=1)
        preds_tuple = preds_tuple[1:]
        preds_valid = [torch.masked_select(pp, valid).unsqueeze(dim=1) for pp in preds_tuple]
        predictions = torch.cat(preds_valid, dim=1)
        assert(predictions.size(1) == num_cats)
        assert(groundtruth.min() >= 0 and groundtruth.max() < num_cats)

        resConfMeter = tnt.meter.ConfusionMeter(num_cats, normalized=False)
        resConfMeter.add(predictions, groundtruth)

        return resConfMeter

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
