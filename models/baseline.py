import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae_resnet import AEResNet
from .utils import tensor2lab
import matplotlib.pyplot as plt


class Baseline(nn.Module):
    def __init__(self, num_classes=21, ngf=64, n_blocks=6):
        super(Baseline, self).__init__()
        self.num_classes = num_classes

        self.encoder = AEResNet(3, num_classes, ngf=ngf, n_blocks=n_blocks, last_layer='softmax')

    def params_to_optimize(self, lrs):
        assert(len(lrs) == 2)
        return [
            {"params": [p for p in self.encoder.parameters() if p.requires_grad], 'lr': lrs[0]},
        ]

    def forward(self, x):
        return self.seg(x), None

    def seg(self, x):
        output, _ = self.encoder(x)
        return output

    def vis(self, data_loader, writer, epoch, num_classes, device):
        for i, batch in enumerate(data_loader):
            if i >= 5:
                break
            image, target = batch['A'].to(device), batch['B'].to(device)
            logits, _ = self.forward(image)

            if epoch == 0:
                # image
                imean = torch.tensor(data_loader.dataset.mean).view(-1,1,1)
                istd = torch.tensor(data_loader.dataset.std).view(-1,1,1)
                image = image[0].detach().cpu() * istd + imean
                image = image.permute(1,2,0)
                # label
                label = target[0].detach()
                label = tensor2lab(label, num_classes, data_loader.dataset.label2color)
                # vis
                fig, axeslist = plt.subplots(ncols=2, nrows=1)
                axeslist[0].imshow(image)
                axeslist[0].set_title('image')
                axeslist[1].imshow(label)
                axeslist[1].set_title('label')
                writer.add_figure(f"gt/img-lab-{i}", fig, global_step=0)

            # logits
            label = logits[0].detach().argmax(0)
            label = tensor2lab(label, num_classes, data_loader.dataset.label2color)
            fig = plt.figure()
            plt.imshow(label)
            writer.add_figure(f"label-pred/lab-epoch{epoch}", fig, global_step=i)
