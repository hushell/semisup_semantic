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
from random import random


class HorseDataset(data.Dataset):
    def __init__(self, root, opt):
        self.root = root
        self.split = 'train' if opt.isTrain else 'test'
        self.heightSize = 32
        self.widthSize = 32
        self.n_classes = 2
        #self.ignore_index = -100
        #self.mean = np.array([128.2095, 124.9667, 106.0276])
        self.mean = [0.5028, 0.4901, 0.4158]
        #self.std = [0.2527, 0.2490, 0.2543]
        self.std = [1.0, 1.0, 1.0]

        # samples
        # TODO: other selected train images
        self.files = os.listdir(root + '/A/')
        self.files = self.files[0:200] if self.split is 'train' else self.files[200:-1]

        # sup indices
        totNum = len(self.files)
        if (opt.sup_portion >= 0 and opt.sup_portion <= 1):
            self.sup_indices = np.random.randint(0, totNum, int(opt.sup_portion * totNum))
        else:
            # sup_portion = 0, 1, ..., 10
            self.sup_indices = np.concatenate([np.arange(i,len(self.files),10)
                                               for i in range(opt.sup_portion)])

        print('==> supervised portion = %.3f' % (float(len(self.sup_indices)) / len(self.files)))

        # transforms
        transform_list = []
        transform_img = tnt.transform.compose([
            lambda x: x.transpose(2,0,1).astype(np.float32),
            lambda x: torch.from_numpy(x).div_(255.0),
            torchvision.transforms.Normalize(mean = self.mean, std = self.std),
        ])

        transform_target = tnt.transform.compose([
            lambda x: x == 255,
            lambda x: x.astype(np.long),
            torch.from_numpy,
            lambda x: x.contiguous(),
        ])

        if opt.isTrain:
            interp_img = cv2.INTER_LINEAR
            interp_target = cv2.INTER_NEAREST

            #if 'scale' in opt.transforms:
            #    target_scale = float(opt.targetSize) / float(max(opt.widthSize, opt.heightSize))
            #    transform_list.append(util.Scale(target_scale, interp_img, interp_target))

            if 'crop' in opt.transforms:
                transform_list.append(util.RandomCrop(crop_width=self.widthSize,
                                                      crop_height=self.heightSize))

            if 'flip' in opt.transforms:
                transform_list.append(util.RandomFlip())

        transform_list.append(util.ImgTargetTransform(img_transform=transform_img,
                                                      target_transform=transform_target))
        self.transform_fun = tnt.transform.compose(transform_list)

        # visualization
        self.label2color = np.array([(0, 0, 0),
                                     (255, 255, 255)])

        self.label2name = np.array(['Background', 'Horse'])

    def name(self):
        return 'HorseDataset'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = os.path.splitext(self.files[index])[0]
        img_path = self.root + '/A/' + img_name + '.jpg'
        lbl_path = self.root + '/B/' + img_name + '.png'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.long)

        #img, lbl = self.transform(img, lbl)
        img, lbl = self.transform_fun((img, lbl))

        return {'A': img, 'B': lbl, 'issup': index in self.sup_indices,
                'A_paths': img_path, 'B_paths': lbl_path}

