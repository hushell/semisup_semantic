import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae_resnet import AEResNet

RESNET_OUTPLANES = 2048


def mask_augment(v_x, drop_rate=0):
    B, _, H, W = v_x.shape[0]
    drop_rate = torch.ones(B, 1, H, W) * (1 - drop_rate)
    mask = torch.bernoulli(drop_rate)
    v_x = v_x * mask.expand_as(v_x)
    return v_x, mask


class SemanticInductiveBias(nn.Module):
    def __init__(self, num_classes=21, drop_rate=0.9):
        super(SemanticInductiveBias, self).__init__()
        self.num_classes = num_classes

        self.encoder = AEResNet(3, num_classes, ngf=64, n_blocks=6, last_layer='softmax')
        self.decoder = AEResNet(num_classes+3, ngf=64, n_blocks=6, last_layer='tanh')

    def forward(self, x):
        logits = self.encoder(x)
        cls_preds = ArgMax.apply(logits) # B x K x H x W

        x_drop, mask = mask_augment(x)
        semantic = torch.cat([cls_preds, x_drop], 1)
        x_hat = self.decoder(semantic)

        return logits, x_hat

    def seg(self, x):
        output = self.encoder(x)
        return output


class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1)
        output = F.one_hot(idx, input.shape[1])
        output = output.permute(0, 3, 1, 2).type_as(input)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Encoder(nn.Module):
    def __init__(self, num_classes, element_dims):
        super(Encoder, self).__init__()
        self.num_classes = num_classes
        self.element_dims = element_dims

        #self.pool = FSPool(output_channels, 20, relaxed=False)
        self.proj_s = nn.Conv1d(RESNET_OUTPLANES, self.element_dims, 1)

    def forward(self, outputs, feats):
        """
        feats: B x F x H x W
        labels: B x H x W
        return: B x F x K
        """
        #cls_set = []
        #for k in range(self.num_classes):
        #    mask = labels == k
        #    sizes = mask.view(mask.shape[0], -1).sum(1)
        #    mask = mask.expand_as(feats)
        #    feat_k = feats * mask
        #    feat_k, _ = self.pool(feat_k, sizes) # B x F
        #    cls_set.append(feat_k)
        #feat_set = torch.stack(cls_set, dim=1) # B x K x F
        B, Fdim, H, W = feats.shape
        assert(Fdim == RESNET_OUTPLANES)

        cls_set = ArgMax.apply(outputs) # B x K x H x W
        sizes = cls_set.sum([2,3]) + 1e-2 # B x K
        feat_set = torch.bmm(feats.view(B, RESNET_OUTPLANES, -1),
                             cls_set.view(B, self.num_classes, -1).transpose(1,2)) # B x F x K
        feat_set = feat_set / sizes.unsqueeze(1)

        feat_set = self.proj_s(feat_set) # B x element_dims x K
        return feat_set


