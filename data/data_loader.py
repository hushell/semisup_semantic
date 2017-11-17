import torch.utils.data
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
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

        indices = np.vstack((unsup_indices, sup_indices)) # matrix: each row contains indices of a batch
        np.random.shuffle(indices) # in place shuffle along axis-0
        return iter(indices.ravel().astype(np.int32))

    def __len__(self):
        return self.len_sup + self.len_unsup

class RepeatRandomSampler(Sampler):
    def __init__(self, subset, fulllen):
        self.subset = torch.from_numpy(subset)
        self.fulllen = fulllen

    def __iter__(self):
        n_sub = len(self.subset)
        n_repeats = self.fulllen // n_sub + 1
        rep_subset = torch.cat([self.subset[torch.randperm(n_sub)] for _ in range(n_repeats)]).long()
        return iter(rep_subset[0:self.fulllen])

    def __len__(self):
        return self.fulllen

class CustomDatasetDataLoader(object):
    def __init__(self, opt):
        self.dataset = CreateDataset(opt)
        batchSize = opt.batchSize if opt.isTrain else 1

        if opt.isTrain:
            if 'unif' in opt.unsup_sampler: # unif, unif_ignore
                my_sampler = SemiSupRandomSampler(self.dataset.unsup, batchSize)
            elif opt.unsup_sampler == 'sep':
                my_sampler = RandomSampler(self.dataset)
                sup_indices = np.where(1 - self.dataset.unsup)[0]
                my_sampler_sup = RepeatRandomSampler(sup_indices, len(self.dataset))
                self.dataloader_sup = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=batchSize,
                    sampler = my_sampler_sup,
                    num_workers=int(opt.nThreads),
                    drop_last=True)
            else:
                raise ValueError("unsup_sampler [%s] not recognized." % opt.unsup_sampler)
        else:
            my_sampler = SequentialSampler(self.dataset)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batchSize,
            #shuffle=True,
            sampler = my_sampler,
            num_workers=int(opt.nThreads),
            drop_last=True if opt.isTrain else False)

    def __iter__(self):
        return self.dataloader.__iter__()

    def __len__(self):
        return len(self.dataset)

    def iter_all(self):
        return self.dataloader.__iter__()

    def iter_sup(self):
        return self.dataloader_sup.__iter__()

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
