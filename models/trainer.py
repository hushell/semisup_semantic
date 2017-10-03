import os
import torch
import itertools
from torch.autograd import Variable

LUT = [(40,0.0001), (100,0.00003), (160,0.00001), (220,0.000003), (240,0.000001)]
LR_DECAY = 0.995 # Applied each epoch "exponential decay"
DECAY_LR_EVERY_N_EPOCHS = 1

class BaseTrainer(object):
    def name(self):
        return 'BaseTrainer'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        if self.gpu_ids:
            self.input_A = torch.cuda.FloatTensor(opt.batchSize, opt.input_nc, opt.heightSize, opt.widthSize)
            self.input_B = torch.cuda.LongTensor(opt.batchSize, opt.output_nc, opt.heightSize, opt.widthSize) # TODO: dim_1 = 1
        else:
            self.input_A = torch.FloatTensor(opt.batchSize, opt.input_nc, opt.heightSize, opt.widthSize)
            self.input_B = torch.LongTensor(opt.batchSize, opt.output_nc, opt.heightSize, opt.widthSize)

        self.models = dict()
        self.optimizers = dict()
        self.lossfuncs = dict()
        self.losses = dict()
        self.fake_pool = dict()

    def _set_model(self, opt):
        ''' G_A (mandatory), D_A, G_B, D_B
        '''
        pass

    def _set_loss(self):
        ''' CE_A, GAN_A, MSE_B, GAN_B
        '''
        pass

    def _set_optim(self, opt, lr_coeffs=None):
        ''' Each network has its own optimizer (currently all the same type)
        '''
        self.old_lr = opt.lr
        self.lr_scheme = opt.lr_scheme
        self.lr_coeffs = lr_coeffs if lr_coeffs is not None else {k:1.0 for k in self.models.keys()}

        for lab, net in self.models.items():
            #parameters = filter(lambda p: p.requires_grad, self.netG_A.parameters())
            parameters = [p for p in net.parameters() if p.requires_grad]
            if opt.optim_method == 'adam':
                self.optimizers[lab] = torch.optim.Adam(itertools.chain(parameters),
                                                        lr=opt.lr*self.lr_coeffs[lab], betas=(opt.beta1, 0.999))
            elif opt.optim_method == 'sgd':
                self.optimizers[lab] = torch.optim.SGD(itertools.chain(parameters),
                                                       lr=opt.lr*self.lr_coeffs[lab], momentum=opt.momentum, weight_decay=opt.weight_decay)
            elif opt.optim_method == 'rmsprop':
                self.optimizers[lab] = torch.optim.RMSprop(itertools.chain(parameters),
                                                           lr=opt.lr*self.lr_coeffs[lab], weight_decay=opt.weight_decay)
            else:
                raise ValueError("Optim_method [%s] not recognized." % opt.optim_method)

    def _set_fake_pool(self):
        pass

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B.long()) # TODO: resize_ unnecessary
        #if len(self.gpu_ids) > 0:
        #    self.input_A = input_A.cuda(self.gpu_ids[0])
        #    self.input_B = input_B.long().cuda(self.gpu_ids[0])
        #else:
        #    self.input_A = input_A
        #    self.input_B = input_B.long()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        unsup = input['unsup']
        assert(unsup.min() == unsup.max())
        self.use_real_B = not bool(unsup[0])

    def get_image_paths(self):
        return self.image_paths

    def forward(self):
        pass

    def backward(self):
        pass

    def optimize_parameters(self):
        # forward
        self.forward()
        # backward
        self.backward()

    # used in test time, no backprop
    def test(self, phase='train'):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_B = self.models['G_A'].forward(self.real_A)

    def update_learning_rate(self, epoch):
        if self.lr_scheme == 'linear':
            if epoch > self.opt.niter:
                lrd = self.opt.lr / self.opt.niter_decay
                lr = self.old_lr - lrd
            else:
                lr = self.opt.lr
        elif self.lr_scheme == 'lut':
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
        elif self.lr_scheme == 'exp':
            lr = self.opt.lr * (LR_DECAY ** (epoch // DECAY_LR_EVERY_N_EPOCHS))
        else:
            raise ValueError("lr scheme [%s] not recognized." % opt.lr_scheme)

        for lab, optim in self.optimizers.items():
            for param_group in optim.param_groups:
                param_group['lr'] = lr * self.lr_coeffs[lab]
            print('===> learning rate of %s: %.10f -> %.10f' % (lab, self.old_lr*self.lr_coeffs[lab], lr*self.lr_coeffs[lab]))

        self.old_lr = lr

    def get_eval_pair(self):
        logits      = self.fake_B.data.cpu().numpy()
        predictions = logits.argmax(1) # NCHW
        groundtruth = self.real_B.data.cpu().numpy()
        return predictions, groundtruth

    def get_current_losses(self):
        pass

    def get_current_visuals(self):
        pass

    def train(self, mode=True):
        for lab in self.models.keys():
            if 'G' in lab:
                self.models[lab].train(mode=mode)

    def on_begin_epoch(self, epoch):
        pass

    def on_end_epoch(self, epoch):
        pass

def CreateTrainer(opt):
    trainer = None
    if opt.loss == 'cross_ent':
        from .cross_entropy_trainer import CrossEntropyTrainer
        trainer = CrossEntropyTrainer(opt)
    elif opt.loss == 'gan_ce':
        from .gan_ce_trainer import GANCrossEntTrainer
        trainer = GANCrossEntTrainer(opt)
    elif opt.loss == 'cycle_gan_ce':
        from .cyclegan_ce_trainer import CycleGANCrossEntTrainer
        trainer = CycleGANCrossEntTrainer(opt)
    elif opt.loss == 'wass_cycle_gan_ce':
        from .wass_cyclegan_ce_trainer import WassCycleGANCrossEntTrainer
        trainer = WassCycleGANCrossEntTrainer(opt)
    #elif opt.loss == 'symm_gan_ce':
    #    from .symmetric_gan_trainer import SymmetricGANCETrainer
    #    trainer = SymmetricGANCETrainer(opt)
    elif opt.loss == 'asp':
        from .amortized_struct_percep_trainer import AmortStructPercepTrainer
        trainer = AmortStructPercepTrainer(opt)
    #elif opt.loss == 'assvm':
    #    from .amortized_struct_svm_trainer import AmortStructSVMTrainer
    #    trainer = AmortStructSVMTrainer(opt)
    #elif opt.loss == 'ebgan':
    #    from .amortized_struct_ebgan_trainer import AmortStructEBGANTrainer
    #    trainer = AmortStructEBGANTrainer(opt)
    #elif opt.loss == 'ace':
    #    from .amortized_cross_ent_trainer import AmortCrossEntTrainer
    #    trainer = AmortCrossEntTrainer(opt)
    else:
        raise ValueError("trainer [%s] not recognized." % opt.loss)
    #trainer.initialize(opt)
    print("===> create_trainer(): [%s] was created" % (trainer.name()))
    return trainer
