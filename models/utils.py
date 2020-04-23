import torch
import torch.nn.functional as F
import numpy as np

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


def mask_augment(v_x, drop_rate=0):
    B, _, H, W = v_x.shape
    drop_rate = torch.ones(B, 1, H, W) * (1 - drop_rate)
    mask = torch.bernoulli(drop_rate).to(v_x.device)
    v_x = v_x * mask.expand_as(v_x)
    return v_x, mask


def tensor2lab(seg_map, n_labs, label2color):
    '''
    seg_map: H x W
    '''
    assert(len(seg_map.shape) == 2)
    seg_map = seg_map.cpu().numpy().astype(np.int32)
    seg_image = label2color[seg_map].astype(np.uint8) # HW3
    return seg_image

