import torch
import torch.nn.functional as F

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


