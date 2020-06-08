import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae_resnet import AEResNet
from .utils import ArgMax, mask_augment, tensor2lab
import matplotlib.pyplot as plt


class SemanticReconstruct(nn.Module):
    def __init__(self, input_nc=3, num_classes=21, drop_rate=0.9, ngf=64, n_blocks=6):
        super(SemanticReconstruct, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.encoder = AEResNet(input_nc, num_classes, ngf=ngf, n_blocks=n_blocks, last_layer='softmax')
        self.decoder = AEResNet(num_classes+input_nc, input_nc, ngf=ngf, n_blocks=n_blocks, last_layer='tanh')

    def params_to_optimize(self, lrs):
        assert(len(lrs) == 2)
        return [
            {"params": [p for p in self.encoder.parameters() if p.requires_grad], 'lr': lrs[0]},
            {"params": [p for p in self.decoder.parameters() if p.requires_grad], 'lr': lrs[1]},
        ]

    def forward(self, x, vis=False):
        logits, _ = self.encoder(x)
        cls_preds = ArgMax.apply(logits) # B x K x H x W

        x_drop, mask = mask_augment(x, self.drop_rate)
        semantic = torch.cat([cls_preds, x_drop], 1)
        x_hat, _ = self.decoder(semantic)

        if vis:
            return logits, x_hat
        else:
            l1 = self.aux_loss(x, x_hat)
            return logits, l1

    def seg(self, x):
        output, _ = self.encoder(x)
        return output

    def aux_loss(self, x, x_hat):
        return F.l1_loss(x_hat, x)

    def vis(self, data_loader, writer, epoch, num_classes, device):
        for i, batch in enumerate(data_loader):
            if i >= 5:
                break
            image, target = batch['A'].to(device), batch['B'].to(device)
            logits, img_hat = self.forward(image, vis=True)

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

            # image_hat
            image = img_hat[0].detach() * 0.5 + 0.5
            image = image.permute(1,2,0).cpu()
            fig = plt.figure()
            plt.imshow(image)
            writer.add_figure(f"image-pred/img-epoch{epoch}", fig, global_step=i)

            # logits
            label = logits[0].detach().argmax(0)
            label = tensor2lab(label, num_classes, data_loader.dataset.label2color)
            fig = plt.figure()
            plt.imshow(label)
            writer.add_figure(f"label-pred/lab-epoch{epoch}", fig, global_step=i)
