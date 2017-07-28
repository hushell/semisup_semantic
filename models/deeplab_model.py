import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
from . import networks
import sys
import imp
import torchnet as tnt
from util.meter import getConfMatrixResults
from .deeplabLargeFOV import Res_Deeplab

POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
RESTORE_FROM = './datasets/MS_DeepLab_resnet_pretrained_COCO_init.pth'

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

class DeeplabModel(BaseModel):
    def name(self):
        return 'DeeplabModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        #size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, opt.heightSize, opt.widthSize)
        self.input_B = self.Tensor(nb, opt.output_nc, opt.heightSize, opt.widthSize)

        # load/define networks, Code (paper): G_A (G)
        assert(opt.which_model_netG == 'deeplab')
        self.netG_A = Res_Deeplab(num_classes = opt.output_nc)
        upsampler = [nn.Upsample(size=(opt.heightSize,opt.widthSize), mode='bilinear'),
                     nn.LogSoftmax()]
        self.upsampler = nn.Sequential(*upsampler)

        # For a small batch size, it is better to keep
        # the statistics of the BN layers (running means and variances)
        # frozen, and to not update the values provided by the pre-trained model.
        # If is_training=True, the statistics will be updated during the training.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.

        saved_state_dict = torch.load(RESTORE_FROM)
        new_params = self.netG_A.state_dict().copy()
        for i in saved_state_dict:
            #Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if opt.output_nc != 21 or i_parts[1] != 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        self.netG_A.load_state_dict(new_params)

        self.netG_A.train()

        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netG_A.cuda(device_id=gpu_ids[0])
            self.upsampler.cuda(device_id=gpu_ids[0])

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.NLLLoss2d()
            if len(self.gpu_ids) > 0:
                self.criterion = self.criterion.cuda()

            # initialize optimizers
            self.old_lr = opt.lr
            self.lr_scheme = opt.lr_scheme

            assert(opt.optim_method == 'sgd')
            self.optimizer_G = optim.SGD([{'params': get_1x_lr_params_NOscale(self.netG_A), 'lr': opt.lr },
                                          {'params': get_10x_lr_params(self.netG_A),        'lr': 10*opt.lr}],
                                          lr = opt.lr, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)

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
        self.fake_B = self.upsampler(self.fake_B)

    def get_image_paths(self):
        return self.image_paths

    def loss_calc(self, out, label):
        loss = 0.0
        for i in range(len(out)):
            loss = loss + self.criterion(out[i],label[i])
        return loss

    def backward_G(self):
        # loss G_A(A)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.fake_B = self.upsampler(self.fake_B)
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
        return OrderedDict([('G_A', G_A), ('lr', self.old_lr)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2lab(self.fake_B.data, self.trainId2color)
        real_B = util.tensor2lab(self.real_B.data, self.trainId2color)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def get_eval_results(self):
        # TODO: diff with SPY and Cityscapes scripts
        logits      = self.fake_B.data.cpu().numpy()
        predictions = logits.argmax(1).squeeze().ravel()
        groundtruth = self.real_B.data.cpu().numpy().ravel()
        nc = logits.shape[1]
        cm = np.zeros((nc, nc))
        for i,n in enumerate(predictions):
            cm[groundtruth[i]][n] += 1
        return cm[1:,1:]

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)

    def update_learning_rate(self, epoch):
        if self.lr_scheme == 'linear':
            if epoch > self.opt.niter:
                lrd = self.opt.lr / self.opt.niter_decay
                lr = self.old_lr - lrd
            else:
                lr = self.opt.lr
        elif self.lr_scheme == 'poly':
            max_epoch = self.opt.niter + self.opt.niter_decay
            lr = self.opt.lr * ((1-float(epoch)/max_epoch)**(POWER))
        else:
            raise ValueError("lr scheme [%s] not recognized." % opt.lr_scheme)

        assert(len(self.optimizer_G.param_groups) == 2)
        self.optimizer_G.param_groups[0]['lr'] = lr
        self.optimizer_G.param_groups[1]['lr'] = lr * 10

        print('===> Update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
