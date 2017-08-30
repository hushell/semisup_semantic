import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data


class CamvidDataset(data.Dataset):
    def __init__(self, root, opt):
        self.root = root
        self.split = opt.phase
        self.heightSize = 360
        self.widthSize = 480
        self.n_classes = 12
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        #self.files = collections.defaultdict(list)
        self.files = os.listdir(root + '/' + self.split)

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

        img, lbl = self.transform(img, lbl)

        return {'A': img, 'B': lbl,
                'A_paths': img_path, 'B_paths': lbl_path}

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float32)
        img -= self.mean
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        #img = torch.from_numpy(img).float()
        #lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        label_colours = self.label2color
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/home/meetshah1995/datasets/segnet/CamVid'
    dst = camvidLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.imshow(dst.decode_segmap(labels.numpy()[i]))
            plt.show()
