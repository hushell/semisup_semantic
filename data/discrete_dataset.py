import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import torch
import torchnet as tnt
import numpy as np


class DiscreteDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        # transforms
        transform_list_img = []
        transform_list_lab = []
        #if opt.resize_or_crop == 'resize_and_crop':
        #    osize = [opt.loadSize, opt.loadSize]
        #    transform_list_img.append(transforms.Scale(osize, Image.BICUBIC))
        #    transform_list_lab.append(transforms.Scale(osize, Image.NEAREST))

        #if opt.isTrain and not opt.no_flip:
        #    transform_list_img.append(transforms.RandomHorizontalFlip())
        #    transform_list_lab.append(transforms.RandomHorizontalFlip())

        #if opt.resize_or_crop != 'no_resize':
        #    transform_list_img.append(transforms.RandomCrop(opt.fineSize))
        #    transform_list_lab.append(transforms.RandomCrop(opt.fineSize))

        #transform_list_img += [transform.Lambda(lambda x: x.transpose(2,0,1).astype(np.float32)),
        #                       transforms.ToTensor(),
        #                       transforms.Normalize((0.485, 0.456, 0.406),
        #                                            (0.229, 0.224, 0.225))]
        transform_list_img += [transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
        transform_list_lab += [transforms.Lambda(lambda x: x.astype(np.int32)),
                               transforms.Lambda(lambda x: torch.from_numpy(x)),
                               transforms.Lambda(lambda x: x.contiguous()),
                               transforms.Lambda(lambda x: x.view(1, x.size(0), x.size(1)))]
        self.transform_img = transforms.Compose(transform_list_img)
        self.transform_lab = transforms.Compose(transform_list_lab)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = np.array(Image.open(A_path))
        B_img = np.array(Image.open(B_path))

        A_img = self.transform_img(A_img)
        B_img = self.transform_lab(B_img)
        #B_img.type(torch.LongTensor)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return min(len(self.A_paths), len(self.B_paths))

    def name(self):
        return 'DiscreteDataset'

