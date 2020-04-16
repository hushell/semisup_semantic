import argparse

#########################################################################
# OPTIONS
parser = argparse.ArgumentParser()

################################
# train settings
################################
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--out_dir', default='./outputs', help='path where to save')
parser.add_argument('--name', type=str, default='ae_mask', help='name of the experiment.')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--seed', type=int, default=123, help='manual seed')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--lrs', type=float, nargs='*', default=[1e-4, 1e-4], help='encoder, decoder')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|lambda2|table')
parser.add_argument('--epochs_decay', type=int, default=80, help='# of epochs to linearly decay lr to zero')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')
parser.add_argument('--sup_portion', type=int, default=0.01, help='portion of supervised, range=0,...,10 or [0,1]')
parser.add_argument('--x_drop', type=float, default=0.99, help='x dropout rate')

################################
# data settings
################################
parser.add_argument('--dataset', type=str, default='horse', help='chooses which dataset is loaded. [cityscapesAB | horse | m2nist]')
parser.add_argument('--widthSize', type=int, default=32, help='crop to this width')
parser.add_argument('--heightSize', type=int, default=32, help='crop to this height')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
parser.add_argument('--ignore_index', type=int, default=-100, help='mask this class without contributing to nll_loss')
parser.add_argument('--transforms', type=str, default='flip', help='crop, flip, resize')
parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')


