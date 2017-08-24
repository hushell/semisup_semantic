import os
import torch
import itertools

LUT = [(40,0.0001), (100,0.00003), (160,0.00001), (220,0.000003), (240,0.000001)]

class BaseTrainer():
    def name(self):
        return 'BaseTrainer'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.heightSize, opt.widthSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.heightSize, opt.widthSize)

        self.models = dict()
        self.optimizers = dict()
        self.lossfuncs = dict()
        self.losses = dict()

    def _set_model(self, opt):
        pass

    def _set_optim(self, opt):
        self.old_lr = opt.lr
        self.lr_scheme = opt.lr_scheme

        for lab, net in self.models.items():
            #parameters = filter(lambda p: p.requires_grad, self.netG_A.parameters())
            parameters = [p for p in net.parameters() if p.requires_grad]
            if opt.optim_method == 'adam':
                self.optimizers[lab] = torch.optim.Adam(itertools.chain(parameters), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim_method == 'sgd':
                self.optimizers[lab] = torch.optim.SGD(itertools.chain(parameters), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            else:
                raise ValueError("Optim_method [%s] not recognized." % opt.optim_method)

    def _set_loss(self):
        pass

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        #if len(self.gpu_ids) > 0:
        #    self.input_A = input_A.cuda(self.gpu_ids[0])
        #    self.input_B = input_B.long().cuda(self.gpu_ids[0])
        #else:
        #    self.input_A = input_A
        #    self.input_B = input_B.long()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_image_paths(self):
        return self.image_paths

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_B = self.models['G_A'].forward(self.real_A)

    def optimize_parameters(self):
        pass

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

        for lab, optim in self.optimizers.items():
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            print('===> learning rate of %s: %f -> %f' % (lab, self.old_lr, lr))

        self.old_lr = lr

    def update_learning_rate(self, epoch):
        lr = self._update_learning_rate(epoch)

    def get_eval_pair(self):
        logits      = self.fake_B.data.cpu().numpy()
        predictions = logits.argmax(1).squeeze() # NCHW
        groundtruth = self.real_B.data.cpu().numpy()
        return predictions, groundtruth

    def get_current_losses(self):
        pass

    def get_current_visuals(self):
        pass

def CreateTrainer(opt):
    trainer = None
    if opt.loss == 'cross_ent':
        from .cross_entropy_trainer import CrossEntropyTrainer
        trainer = CrossEntropyTrainer(opt)
    else:
        raise ValueError("trainer [%s] not recognized." % opt.loss)
    #trainer.initialize(opt)
    print("===> create_trainer(): [%s] was created" % (trainer.name()))
    return trainer
