from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import matplotlib.pyplot as plt

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0]
    #image_numpy = np.transpose(image_numpy, (1, 2, 0)) # CHW -> HWC
    #image_numpy = image_numpy * 255.0
    return image_numpy.astype(imtype)

def tensor2lab(seg_map, n_labs=None, label2color=None):
    '''
    seg_map: H x W
    '''
    assert(len(seg_map.shape) == 2)
    assert(n_labs is not None or label2color is not None)
    if label2color is None:
        label2color = plt.cm.jet(np.linspace(0,1,n_labs), bytes=True)[:,0:3]
    seg_map = seg_map.cpu().numpy().astype(np.int32)
    seg_image = label2color[seg_map].astype(np.uint8) # HW3
    #return seg_image.transpose((2,0,1)) # 3HW
    return seg_image

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_(x, dim=1):
    '''
    Projects points to a sphere inplace.
    '''
    x.div_(x.norm(2, dim=dim, keepdim=True).expand_as(x))


def normalize(x, dim=1):
    '''
    Projects points to a sphere.
    '''
    return x.div(x.norm(2, dim=dim, keepdim=True).expand_as(x))


def var(x, dim=0):
    '''
    Calculates variance.
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


def partial_sum(v, keep_dims=[]):
    """Sums variable or tensor along all dimensions except those specified
    in `keep_dims`"""
    if len(keep_dims) == 0:
        return v.sum()
    else:
        keep_dims = sorted(keep_dims)
        drop_dims = list(set(range(v.dim())) - set(keep_dims))
        result = v.permute(*(keep_dims + drop_dims))
        size = result.size()[:len(keep_dims)] + (-1,)
        return result.contiguous().view(size).sum(-1)


