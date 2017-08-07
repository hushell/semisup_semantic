import torch.utils.data

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader(opt)
    print(data_loader.name())
    return data_loader

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.nThreads),
            drop_last=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

    def name(self):
        return 'CustomDatasetDataLoader'

def CreateDataset(opt):
    dataset = None
    data_path = get_data_path(opt.dataset)
    if opt.dataset == 'pascal':
        from data.pascal_voc_dataset import PascalVOCDataset
        dataset = PascalVOCDataset(data_path, is_transform=True, img_size=(opt.heightSize, opt.widthSize))
    elif opt.dataset == 'cityscapesAB':
        from data.cityscapesAB_dataset import CityscapesDataset
        dataset = CityscapesDataset(data_path, opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset)

    print("dataset [%s] was created" % (dataset.name()))
    return dataset

def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    import json
    data = json.load(open('data/config.json'))
    return data[name]['data_path']
