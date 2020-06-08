import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae_resnet import AEResNet
from .utils import ArgMax, mask_augment, tensor2lab
import matplotlib.pyplot as plt


class SemanticConsistency(nn.Module):
    def __init__(self, input_nc=3, num_classes=21, element_dims=128, drop_rate=0.9,
                 ngf=64, n_blocks=6, H=64, W=64):
        super(SemanticConsistency, self).__init__()
        self.num_classes = num_classes
        self.element_dims = element_dims
        self.drop_rate = drop_rate

        self.encoder = AEResNet(input_nc, num_classes, ngf=ngf, n_blocks=n_blocks, last_layer='softmax')
        #self.spool = SetPool(num_classes, ngf, element_dims)
        #self.decoder = SetDecoder(element_dims, H, W)

    def params_to_optimize(self, lrs):
        assert(len(lrs) == 2)
        return [
            {"params": [p for p in self.encoder.parameters() if p.requires_grad], 'lr': lrs[0]},
            #{"params": [p for p in self.spool.parameters() if p.requires_grad], 'lr': lrs[1]},
        ]

    def forward(self, x, vis=False):
        logits, feats = self.encoder(x)
        #cls_feats = self.spool(logits, feats)

        x_drop, mask = mask_augment(x, self.drop_rate) # TODO: rand drop rate
        logits_msk, feats_msk = self.encoder(x_drop)
        #cls_feats_msk = self.spool(logits_msk, feats_msk)

        if vis:
            return logits, logits_msk, x_drop
        else:
            #consistency = self.aux_loss(cls_feats, cls_feats_msk)
            consistency = self.aux_loss(logits, logits_msk)
            return logits, consistency

    def seg(self, x):
        output, _ = self.encoder(x)
        return output

    def aux_loss(self, *args, **kwargs):
        #cls_feats, cls_feats_msk = args
        #return F.mse_loss(cls_feats, cls_feats_msk)
        logits, logits_msk = args
        return F.cross_entropy(logits_msk, logits.detach().argmax(1))

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

class SetPool(nn.Module):
    def __init__(self, num_classes, ngf, element_dims):
        super(SetPool, self).__init__()
        self.num_classes = num_classes
        self.ngf = ngf
        self.element_dims = element_dims

        #self.pool = FSPool(output_channels, 20, relaxed=False)
        #self.proj_s = nn.Conv1d(ngf, self.element_dims, 1)
        self.bn = nn.BatchNorm1d(ngf)

    def forward(self, outputs, feats):
        """
        Find the average feature for each class

        feats: B x F x H x W
        labels: B x H x W
        return: B x F x K
        """
        B = feats.shape[0]

        cls_set = ArgMax.apply(outputs) # B x K x H x W
        feat_set = torch.bmm(feats.view(B, self.ngf, -1),
                             cls_set.view(B, self.num_classes, -1).transpose(1,2)) # B x F x K

        feat_set = self.bn(feat_set)
        #feat_set = self.proj_s(feat_set) # B x element_dims x K
        return feat_set


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()

        self.ngf = 64 # was 256
        g_ksize = 4
        self.proj = nn.Linear(input_dim, self.ngf * 4 * 4 * 4)
        self.bn0 = nn.BatchNorm1d(self.ngf * 4 * 4 * 4)

        self.dconv1 = nn.ConvTranspose2d(self.ngf*4,self.ngf*2, g_ksize,
                            stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ngf*2)

        self.dconv2 = nn.ConvTranspose2d(self.ngf*2, self.ngf, g_ksize,
                            stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ngf)

        self.dconv3 = nn.ConvTranspose2d(self.ngf, 3, g_ksize,
                            stride=4, padding=0, bias=False)

    def forward(self, z, c=None):
        out = F.relu(self.bn0(self.proj(z)).view(-1, self.ngf*4, 4, 4))
        out = F.relu(self.bn1(self.dconv1(out)))
        out = F.relu(self.bn2(self.dconv2(out)))
        out =  self.dconv3(out)
        return out


class SetDecoder(nn.Module):
    def __init__(self, element_dims, H, W):
        super(SetDecoder, self).__init__()
        self.vec_decoder = Decoder(element_dims)
        self.H, self.W = H, W

    def forward(self, x_set):
        batch_size, element_dims, set_size = x_set.shape

        x = x_set.transpose(1,2).reshape(-1, element_dims)
        generated = self.vec_decoder(x) # decode every element in batch x set
        generated = generated.reshape(batch_size, set_size, 3, self.H, self.W)

        attention = torch.softmax(generated, dim=1)
        generated_set = torch.sigmoid(generated)

        generated_set = generated_set*attention
        img_hat = generated_set.sum(dim=1).clamp(0,1)

        return img_hat, generated_set

