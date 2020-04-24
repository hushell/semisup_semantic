import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae_resnet import AEResNet
from .utils import ArgMax, mask_augment, tensor2lab
import matplotlib.pyplot as plt


class SemanticSelfSupConsistency(nn.Module):
    def __init__(self, num_classes=21, element_dims=128, drop_rate=0.9,
                 ngf=64, n_blocks=6, H=64, W=64):
        super(SemanticSelfSupConsistency, self).__init__()
        self.num_classes = num_classes
        self.element_dims = element_dims
        self.drop_rate = drop_rate

        self.encoder = AEResNet(3, num_classes, ngf=ngf, n_blocks=n_blocks, last_layer='softmax')

        model = [nn.Conv2d(ngf, 2, kernel_size=7, padding=3)]
        model += [nn.LogSoftmax(dim=1)]
        self.ss_classifier = nn.Sequential(*model)

    def params_to_optimize(self, lrs):
        assert(len(lrs) == 2)
        return [
            {"params": [p for p in self.encoder.parameters() if p.requires_grad], 'lr': lrs[0]},
            {"params": [p for p in self.ss_classifier.parameters() if p.requires_grad], 'lr': lrs[1]},
        ]

    def forward(self, x, vis=False):
        logits, feats = self.encoder(x)

        x_drop, mask = mask_augment(x, self.drop_rate) # TODO: rand drop rate
        logits_msk, feats_msk = self.encoder(x_drop)

        if vis:
            return logits, logits_msk, x_drop
        else:
            consistency = self.consist_loss(logits, logits_msk)
            selfsup = self.selfsup_loss(feats_msk, mask)
            return logits, {'consist':consistency, 'selfsup':selfsup * 0.05}

    def seg(self, x):
        output, _ = self.encoder(x)
        return output

    def consist_loss(self, logits, logits_msk):
        return F.cross_entropy(logits_msk, logits.detach().argmax(1))

    def selfsup_loss(self, feats_msk, mask):
        mask_pred = self.ss_classifier(feats_msk)
        return F.cross_entropy(mask_pred, mask.detach().squeeze(1).long())

    def vis(self, data_loader, writer, epoch, num_classes, device):
        for i, batch in enumerate(data_loader):
            if i >= 5:
                break
            image, target = batch['A'].to(device), batch['B'].to(device)
            logits, logits_msk, img_msk = self.forward(image, vis=True)

            if epoch == 0:
                # image
                imean = torch.tensor(data_loader.dataset.mean).view(-1,1,1)
                istd = torch.tensor(data_loader.dataset.std).view(-1,1,1)
                image = image[0].detach().cpu() * istd + imean
                image = image.permute(1,2,0)
                # image masked
                img_msk = img_msk[0].detach().cpu() * istd + imean
                img_msk = img_msk.permute(1,2,0)
                # label
                label = target[0].detach()
                label = tensor2lab(label, num_classes, data_loader.dataset.label2color)
                # vis
                fig, axeslist = plt.subplots(ncols=3, nrows=1)
                axeslist[0].imshow(image)
                axeslist[0].set_title('image')
                axeslist[1].imshow(label)
                axeslist[1].set_title('label')
                axeslist[2].imshow(img_msk)
                axeslist[2].set_title('masked image')
                writer.add_figure(f"gt/img-lab-{i}", fig, global_step=0)

            # logits
            label = logits[0].detach().argmax(0)
            label = tensor2lab(label, num_classes, data_loader.dataset.label2color)
            # masked logits
            label_msk = logits_msk[0].detach().argmax(0)
            label_msk = tensor2lab(label_msk, num_classes, data_loader.dataset.label2color)
            # vis
            fig, axeslist = plt.subplots(ncols=2, nrows=1)
            axeslist[0].imshow(label)
            axeslist[0].set_title('clean')
            axeslist[1].imshow(label_msk)
            axeslist[1].set_title('masked')
            writer.add_figure(f"label-pred/lab-epoch{epoch}", fig, global_step=i)

