from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--optim_method', type=str, default='adam', help='adam, sgd')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_scheme', type=str, default='linear', help='linear, lut')
        self.parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--gt_noise', action='store_true', help='use scaled gt')
        self.parser.add_argument('--no_save', action='store_true', help='do not save states')
        self.parser.add_argument('--unsup_portion', type=int, default=0, help='portion of unsupervised, range=0,...,10')
        self.parser.add_argument('--portion_total', type=int, default=10, help='total portion of unsupervised, e.g., 10')
        self.parser.add_argument('--unsup_ignore', action='store_true', help='ignore all unsup examples')
        self.isTrain = True
