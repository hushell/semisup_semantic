import argparse

#########################################################################
# OPTIONS
parser = argparse.ArgumentParser()
################################
# data settings
################################
parser.add_argument('--dataset', type=str, default='horse', help='chooses which dataset is loaded. [cityscapesAB | pascal | camvid]')
parser.add_argument('--widthSize', type=int, default=32, help='crop to this width')
parser.add_argument('--heightSize', type=int, default=32, help='crop to this height')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
parser.add_argument('--ignore_index', type=int, default=-100, help='mask this class without contributing to nll_loss')
parser.add_argument('--sup_portion', type=int, default=9, help='portion of unsupervised, range=0,...,10')
parser.add_argument('--transforms', type=str, default='flip', help='crop, flip, resize')

################################
# train settings
################################
parser.add_argument('--name', type=str, default='ae_mask', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--output-dir', default='./outputs', help='path where to save')
parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')
parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lrs', type=float, nargs='*', default=[1e-4, 1e-4], help='encoder, decoder')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')


################################
# model settings
################################
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--z_nc', type=int, default=40, help='# of output image channels')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--n_layers_D', type=int, default=3, help='')
parser.add_argument('--x_drop', type=float, default=0.9, help='x dropout rate')

################################
# external
################################
parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--port', type=int, default=8097, help='port of visdom')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

