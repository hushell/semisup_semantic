import os
import collections
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import numpy as np
import scipy.misc as m
import util
import cv2


class CamvidDataset(data.Dataset):
    def __init__(self, root, opt):
        self.root = root
        self.split = opt.phase
        self.heightSize = 360
        self.widthSize = 480
        self.n_classes = 12
        self.ignore_index = 11
        #self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
        #self.std = [0.27413549931506, 0.28506257482912, 0.28284674400252]
        self.std = [1.0, 1.0, 1.0]

        # samples
        self.files = np.array( os.listdir(root + '/' + self.split) )
        self.unsup = np.zeros(self.__len__(), dtype=np.int32)

        if opt.isTrain and opt.unsup_portion > 0:
            assert(opt.unsup_portion <= opt.portion_total) # e.g., unsup_portion=0: no unsup; unsup_portion=portion_total=10: all unsup
            tmp = np.concatenate([np.arange(i,self.__len__(),opt.portion_total) for i in range(opt.unsup_portion)])
            self.unsup[tmp] = 1
            print('==> unsupervised portion = %.3f' % (float(sum(self.unsup)) / self.__len__()))

        # transforms
        transform_list = []
        transform_img = tnt.transform.compose([
            #lambda x: x[:,:,::-1], # RGB->BGR
            lambda x: x.transpose(2,0,1).astype(np.float32),
            lambda x: torch.from_numpy(x).div_(255.0),
            torchvision.transforms.Normalize(mean = self.mean, std = self.std),
        ])

        transform_target = tnt.transform.compose([
            lambda x: x.astype(np.int32),
            torch.from_numpy,
            lambda x: x.contiguous(),
            #lambda x: x.view(1,x.size(0), x.size(1)), # HxW -> 1xHxW
        ])

        if opt.isTrain:
            interp_img = cv2.INTER_LINEAR
            interp_target = cv2.INTER_NEAREST

            if 'resize' in opt.resize_or_crop:
                target_scale = float(opt.targetSize) / float(max(opt.widthSize, opt.heightSize))
                transform_list.append(util.Scale(target_scale, interp_img, interp_target))

            if 'crop' in opt.resize_or_crop:
                transform_list.append(util.RandomCrop(crop_width=opt.widthSize, crop_height=opt.heightSize))

            if not opt.no_flip:
                transform_list.append(util.RandomFlip())

        transform_list.append(util.ImgTargetTransform(img_transform=transform_img, target_transform=transform_target))
        self.transform_fun = tnt.transform.compose(transform_list)

        # visualization
        self.label2color = np.array([(128, 128, 128),
                                     (128, 0, 0),
                                     (192, 192, 128),
                                     (128, 64, 128),
                                     (0, 0, 192),
                                     (128, 128, 0),
                                     (192, 128, 128),
                                     (64, 64, 128),
                                     (64, 0, 128),
                                     (64, 64, 0),
                                     (0, 128, 192),
                                     (0, 0, 0)])

        self.label2name = np.array(['Sky', 'Building', 'Column-Pole', 'Road',
                                    'Sidewalk', 'Tree', 'Sign-Symbol', 'Fence', 'Car', 'Pedestrain',
                                    'Bicyclist', 'Void'])

    def name(self):
        return 'CamvidDataset'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = self.root + '/' + self.split + '/' + img_name
        lbl_path = self.root + '/' + self.split + 'annot/' + img_name

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        img, lbl = self.transform_fun((img, lbl))

        return {'A': img, 'B': lbl, 'unsup': self.unsup[index],
                'A_paths': img_path, 'B_paths': lbl_path}

