import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps, ImageFilter
from matplotlib import cm


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        img = sample['A']
        label = sample['B']
        img = np.array(img).astype(np.float32)
        label = np.array(label).astype(np.float32)
        mean = self.mean.reshape(-1, 1, 1)
        std = self.std.reshape(-1, 1, 1)
        img -= mean
        img /= std

        return {'A': img,
                'B': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img = sample['A']
        mask = sample['B']

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long().squeeze()

        return {'A': img,
                'B': mask}


class MODISDataset(Dataset):
    heightSize = 48
    widthSize = 72
    n_classes = 7
    mean=[4.8003e+00, 2.5270e+03, 4.4159e+02, 2.4647e+00, 1.9367e+04, 7.8704e+00],
    std=[4.5098e+00, 9.4051e+02, 9.4881e+01, 1.4661e+00, 1.4378e+04, 5.9816e+00]

    def __init__(self, root, opt):
        super().__init__()
        self.train = opt.isTrain
        self.modis_dic = np.load(os.path.join(root, 'dataset.npy'), allow_pickle='True').item()
        self.X_train, self.y_train_raw, self.X_test, self.y_test_raw = self.data_split(self.modis_dic)

        def discretize(y_raw):
            y = np.zeros_like(y_raw, dtype=np.int)
            y[y_raw == -1] = -1 # will be ignored in CE loss
            y[y_raw == 0] = 0
            y[(y_raw > 0) & (y_raw < 5)] = 1
            y[(y_raw >= 5) & (y_raw < 17)] = 2
            y[(y_raw >= 17) & (y_raw < 38)] = 3
            y[(y_raw >= 38) & (y_raw < 75)] = 4
            y[(y_raw >= 75) & (y_raw < 100)] = 5
            y[(y_raw >= 100)] = 6
            return y

        self.y_train, self.y_test = discretize(self.y_train_raw), discretize(self.y_test_raw)

        # sup indices
        self.sup_indices = range(len(self.y_train))

        # visualization
        self.label2color = cm.get_cmap('viridis')(np.linspace(0.0, 1.0, self.n_classes))[:,:3]

        self.label2name = np.array(['no rain', '(0, 5)', '[5, 17)', '[17, 38)', '[38, 75)',
                                    '[75, 100)', '[100, infty)'])

        self.transform = transforms.Compose([
            Normalize(
                mean=[4.8003e+00, 2.5270e+03, 4.4159e+02, 2.4647e+00, 1.9367e+04, 7.8704e+00],
                std=[4.5098e+00, 9.4051e+02, 9.4881e+01, 1.4661e+00, 1.4378e+04, 5.9816e+00]
            ),
            ToTensor()
        ])


    def __getitem__(self, idx):
        if self.train:
            image = self.X_train[idx]
            target = self.y_train[idx]
        else:
            image = self.X_test[idx]
            target = self.y_test[idx]
        sample = {'A': image, 'B': target}
        sample = self.transform(sample)
        sample['issup'] = True
        return sample


    def __len__(self):
        return len(self.X_train[0])

    def name(self):
        return 'MODISDataset'


    def data_split(self, modis_dic):
        data_set = np.array(list(modis_dic.values()))
        day_idx = np.array(list(modis_dic.keys()))
        images = data_set[:, :-1]
        targets = data_set[:, -1].reshape(-1, 1, 48, 72)

        day_idx = [int(i/31) for i in day_idx]

        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=0, stratify=day_idx)

        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from collections import OrderedDict
    opt = OrderedDict()
    opt.isTrain = True
    modis_train = MODISDataset('../datasets/modis', opt)
    dataloader = DataLoader(modis_train, batch_size=8, shuffle=True, num_workers=4)

    for i, sample in enumerate(dataloader):
        image, target = sample['A'], sample['B']
        print(image.shape)
        print(target.shape)
        break
