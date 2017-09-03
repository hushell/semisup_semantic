import torch.utils.data
from torch.utils.data.sampler import Sampler, RandomSampler
import numpy as np

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader(opt)
    return data_loader

class SemiSupRandomSampler(Sampler):
    def __init__(self, unsup, batchSize):
        self.unsup = unsup # int ndarray only {0,1}
        self.batchSize = batchSize

    def __iter__(self):
        unsup_indices = np.random.permutation( np.where(self.unsup)[0] ) # indices from bool
        sup_indices = np.random.permutation( np.where(1 - self.unsup)[0] )

        self.len_unsup = len(unsup_indices) - len(unsup_indices) % self.batchSize # mod by batchSize
        self.len_sup = len(sup_indices) - len(sup_indices) % self.batchSize
        unsup_indices = unsup_indices[0:self.len_unsup].reshape((-1,self.batchSize))
        sup_indices = sup_indices[0:self.len_sup].reshape((-1,self.batchSize))

        indices = np.vstack((unsup_indices, sup_indices))
        np.random.shuffle(indices) # in place shuffle along axis-0
        return iter(indices.ravel().astype(np.int32))

    def __len__(self):
        return self.len_sup + self.len_unsup

class CustomDatasetDataLoader(object):
    def __init__(self, opt):
        self.dataset = CreateDataset(opt)
        my_sampler = SemiSupRandomSampler(self.dataset.unsup, opt.batchSize) \
                        if opt.unsup_portion > 0 else RandomSampler
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            #shuffle=True,
            sampler = my_sampler,
            num_workers=int(opt.nThreads),
            drop_last=True)

    def __iter__(self):
        return self.dataloader.__iter__()

    def __len__(self):
        return len(self.dataset)

    def name(self):
        return 'CustomDatasetDataLoader'

    def update_opt(self, opt):
        if hasattr(self.dataset, 'n_classes'):
            opt.output_nc = self.dataset.n_classes
        if hasattr(self.dataset, 'heightSize'):
            opt.heightSize = self.dataset.heightSize
        if hasattr(self.dataset, 'widthSize'):
            opt.widthSize = self.dataset.widthSize
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
