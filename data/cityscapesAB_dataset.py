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
import imp


class CityscapesABDataset(data.Dataset):
    def __init__(self, dataroot, opt):
        self.opt = opt
        self.root = dataroot

        phase = 'train' if opt.isTrain else 'val'
        self.dir_A = os.path.join(dataroot, phase + 'A')
        self.dir_B = os.path.join(dataroot, phase + 'B')

        self.heightSize = 256
        self.widthSize = 256
        self.n_classes = 20
        self.ignore_index = 0
        self.mean = [ 0.485, 0.456, 0.406 ]
        #self.std = [ 0.229, 0.224, 0.225 ]
        self.std = [ 1.0, 1.0, 1.0 ]

        # files
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        # sup indices
        totNum = self.__len__()
        if (opt.sup_portion >= 0 and opt.sup_portion <= 1):
            self.sup_indices = np.random.randint(0, totNum, int(opt.sup_portion * totNum))
        else:
            # sup_portion = 0, 1, ..., 10
            self.sup_indices = np.concatenate([np.arange(i, totNum, 10)
                                               for i in range(opt.sup_portion)])

        print('==> supervised portion = %.3f' % (float(len(self.sup_indices)) / totNum))

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

            #if 'scale' in opt.transforms:
            #    target_scale = float(opt.targetSize) / float(max(opt.widthSize, opt.heightSize))
            #    transform_list.append(util.Scale(target_scale, interp_img, interp_target))

            if 'crop' in opt.transforms:
                transform_list.append(util.RandomCrop(crop_width=self.widthSize,
                                                      crop_height=self.heightSize))

            if 'flip' in opt.transforms:
                transform_list.append(util.RandomFlip())

        transform_list.append(util.ImgTargetTransform(img_transform=transform_img, target_transform=transform_target))
        self.transform_fun = tnt.transform.compose(transform_list)

        # visualization
        cslabels = imp.load_source("",'%s/labels.py' % self.root)
        label2trainId = np.asarray([(1+label.trainId) if label.trainId < 255 else 0 for label in cslabels.labels], dtype=np.float32)
        label2color = np.asarray([(label.color) for label in cslabels.labels], dtype=np.uint8)
        num_cats      = 1+19 # the first extra category is for the pixels with missing category
        trainId2labelId = np.ndarray([num_cats], dtype=np.int32)
        trainId2labelId.fill(-1)
        for labelId in range(len(cslabels.labels)):
            trainId = int(label2trainId[labelId])
            if trainId2labelId[trainId] == -1:
                trainId2labelId[trainId] = labelId
        self.label2color = label2color[trainId2labelId] # ndarray 20x3
        clsNames = np.asarray([label.name for label in cslabels.labels], dtype=np.str)
        self.label2name = clsNames[trainId2labelId] # ndarray 20


    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = np.array(Image.open(A_path), dtype=np.float32)
        B_img = np.array(Image.open(B_path), dtype=np.int32)

        A_img, B_img = self.transform_fun((A_img, B_img))

        return {'A': A_img, 'B': B_img, 'issup': index in self.sup_indices,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return min(len(self.A_paths), len(self.B_paths))

    def name(self):
        return 'CityscapesABDataset'
