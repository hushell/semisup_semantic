import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
import numpy as np


class CustomDatasetDataLoader(object):
    def __init__(self, opt, istrain=False, suponly=False):
        opt.isTrain = istrain
        self.dataset = CreateDataset(opt)
        batchSize = opt.batchSize if istrain else 1
        self.batchSize = batchSize

        if istrain and suponly:
            my_sampler = SubsetRandomSampler(self.dataset.sup_indices)
        else:
            my_sampler = None

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=istrain and not suponly,
            sampler=my_sampler,
            num_workers=int(opt.nThreads),
            drop_last=False)

    def __iter__(self):
        return self.dataloader.__iter__()

    def __len__(self):
        return self.dataloader.__len__()

    def name(self):
        return 'CustomDatasetDataLoader'

    def update_opt(self, opt):
        if hasattr(self.dataset, 'n_classes'):
            opt.output_nc = self.dataset.n_classes
        if hasattr(self.dataset, 'ignore_index'):
            opt.ignore_index = self.dataset.ignore_index
        #if hasattr(self.dataset, 'heightSize'):
        #    opt.heightSize = self.dataset.heightSize
        #if hasattr(self.dataset, 'widthSize'):
        #    opt.widthSize = self.dataset.widthSize
        return opt


def CreateDataset(opt):
    dataset = None
    data_path = get_data_path(opt.dataset)
    if opt.dataset == 'pascal':
        from .pascal_voc_dataset import PascalVOCDataset
        dataset = PascalVOCDataset(data_path, is_transform=True, img_size=(opt.heightSize, opt.widthSize))
    elif opt.dataset == 'cityscapesAB':
        from .cityscapesAB_dataset import CityscapesABDataset
        dataset = CityscapesABDataset(data_path, opt)
    elif opt.dataset == 'camvid':
        from .camvid_dataset import CamvidDataset
        dataset = CamvidDataset(data_path, opt)
    elif opt.dataset == 'horse':
        from .horse_dataset import HorseDataset
        dataset = HorseDataset(data_path, opt)
    elif opt.dataset == 'm2nist':
        from .m2nist_dataset import M2NISTDataset
        dataset = M2NISTDataset(data_path, opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset)

    print("===> dataset [%s] was created" % (dataset.name()))
    return dataset


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    import json
    data = json.load(open('data/config.json'))
    return data[name]['data_path']


class InfiniteDataLoader(object):
    """Allow to load sample infinitely"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = None

    def next(self):
        try:
            data = self.data_iter.next()
        except Exception:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = self.data_iter.next()
            #print('*** infi_data_loader starts over.')

        return data

    def __len__(self):
        return len(self.dataloader)
