import os.path
from data.image_folder import make_dataset
from PIL import Image
import torch
import torch.utils.data as data
import torchnet as tnt
import numpy as np
import util
import cv2
import torchvision


class CityscapesABDataset(data.Dataset):
    def __init__(self, dataroot, opt):
        self.opt = opt
        self.root = dataroot
        self.dir_A = os.path.join(dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        # transforms
        transform_list = []
        transform_img = tnt.transform.compose([
            lambda x: x.transpose(2,0,1).astype(np.float32),
            lambda x: torch.from_numpy(x).div_(255.0),
            torchvision.transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                             std = [ 0.229, 0.224, 0.225 ]),
        ])

        transform_target = tnt.transform.compose([
            lambda x: x.astype(np.int32),
            torch.from_numpy,
            lambda x: x.contiguous(),
            #lambda x: x.view(1,x.size(0), x.size(1)),
        ])

        interp_img = cv2.INTER_LINEAR
        interp_target = cv2.INTER_NEAREST
        if opt.isTrain:
            if opt.resize_or_crop == 'resize_and_crop':
                scale = float(opt.loadSize) / float(max(opt.widthSize, opt.heightSize))
                transform_list.append(util.Scale(scale, interp_img, interp_target))

            if not opt.no_flip:
                transform_list.append(util.RandomFlip())

            if opt.resize_or_crop != 'no_resize':
                transform_list.append(util.RandomCrop(crop_width=opt.widthSize, crop_height=opt.heightSize))

            transform_list.append(util.ImgTargetTransform(img_transform=transform_img, target_transform=transform_target))
            self.transform_fun = tnt.transform.compose(transform_list)
        else:
            #target_scale = opt.targetScale if hasattr(opt, 'target_scale') else 1.0
            target_scale = 1.0
            self.transform_fun = tnt.transform.compose([
                util.Scale(scale=target_scale, interp_img=interp_img, interp_target=interp_target),
                util.ImgTargetTransform(img_transform=transform_img, target_transform=transform_target),
            ])

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = np.array(Image.open(A_path), dtype=np.float32)
        B_img = np.array(Image.open(B_path), dtype=np.int32)

        A_img, B_img = self.transform_fun((A_img, B_img))

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return min(len(self.A_paths), len(self.B_paths))

    def name(self):
        return 'CityscapesABDataset'
