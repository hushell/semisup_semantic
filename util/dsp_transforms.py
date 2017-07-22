# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:53:10 2017

@author: spyros
"""
from __future__ import division
import random
import numpy as np
import cv2

def img_scale(img, scale, interpolation):
    assert(isinstance(img,np.ndarray))
    width, height = img.shape[1], img.shape[0]
    scale = float(scale)
    if scale != 1.0:
        owidth  = int(float(width)  * scale)
        oheight = int(float(height) * scale)
        img = cv2.resize(img, dsize=(owidth, oheight), interpolation=interpolation)

    return img

def img_resize(img, new_width, new_height, interpolation):
    assert(isinstance(img,np.ndarray))
    width, height = img.shape[1], img.shape[0]
    if width != new_width or height != new_height:
        img = cv2.resize(img, dsize=(new_width, new_height), interpolation=interpolation)

    return img

def sample_scale_sep(sample, scale_img, interp_img, scale_target, interp_target):
    img, target = sample[:2]
    img = img_scale(img, scale_img, interp_img)
    target = img_scale(target, scale_target, interp_target)

    return (img, target) + sample[2:]

def sample_scale(sample, scale, img_interp, target_interp):
    img, target = sample[:2]
    img = img_scale(img, scale, img_interp)
    target = img_scale(target, scale, target_interp)

    return (img, target) + sample[2:]

def sample_flip(sample):
    img, target = sample[:2]
    assert(isinstance(img,np.ndarray))
    assert(isinstance(target,np.ndarray))
    img = cv2.flip(img, 1).reshape(img.shape)
    target = cv2.flip(target, 1).reshape(target.shape)
    return (img, target) + sample[2:]

def sample_crop(sample, crop_loc):
    img, target = sample[:2]
    assert(isinstance(img,np.ndarray))
    assert(isinstance(img,np.ndarray))
    assert(img.shape[1] == target.shape[1])
    assert(img.shape[0] == target.shape[0])
    width, height = img.shape[1], img.shape[0]
    x0, y0, x1, y1 = crop_loc
    if not (x0==0 and x1==width and y0==0 and y1==height):
        img = img[y0:y1,x0:x1]
        target = target[y0:y1,x0:x1]

    return (img, target) + sample[2:]


class RandomFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample = sample_flip(sample)
        return sample

class Scale(object):
    def __init__(self, scale, interp_img=cv2.INTER_LINEAR, interp_target=cv2.INTER_NEAREST):
        self.scale = scale
        self.interp_img = interp_img
        self.interp_target = interp_target

    def __call__(self, sample):
        return sample_scale(sample, self.scale, self.interp_img, self.interp_target)

class ScaleSep(object):
    def __init__(self, scale_img, scale_target, interp_img=cv2.INTER_LINEAR, interp_target=cv2.INTER_NEAREST):
        self.scale_img     = scale_img
        self.scale_target  = scale_target
        self.interp_img    = interp_img
        self.interp_target = interp_target

    def __call__(self, sample):
        return sample_scale_sep(sample, self.scale_img, self.interp_img, self.scale_target, self.interp_target)

class RandomScale(object):
    def __init__(self, min_scale, max_scale, interp_img=cv2.INTER_LINEAR, interp_target=cv2.INTER_NEAREST):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interp_img = interp_img
        self.interp_target = interp_target

    def __call__(self, sample):
        scale = random.uniform(self.min_scale, self.max_scale)
        return sample_scale(sample, scale, self.interp_img, self.interp_target)


class RandomCrop(object):
    def __init__(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __call__(self, sample):
        img, target = sample[:2]
        assert(isinstance(img,np.ndarray))
        assert(isinstance(img,np.ndarray))
        assert(img.shape[1] == target.shape[1])
        assert(img.shape[0] == target.shape[0])
        width, height = img.shape[1], img.shape[0]
        x0 = random.randint(0, width  - self.crop_width)
        y0 = random.randint(0, height - self.crop_height)
        x1 = x0 + self.crop_width
        y1 = y0 + self.crop_height
        crop_loc = (x0, y0, x1, y1)
        return sample_crop(sample, crop_loc)

class ImgTargetTransform(object):
    def __init__(self, img_transform, target_transform):
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __call__(self, sample):
        img, target = sample[:2]
        if self.img_transform != None:
            img = self.img_transform(img)

        if self.target_transform != None:
            target = self.target_transform(target)

        return (img, target) + sample[2:]

class ToDict(object):
    def __call__(self, sample):
        img, target = sample[:2]
        return {'input':img, 'target':target}
