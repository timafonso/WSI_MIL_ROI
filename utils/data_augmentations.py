from torchvision import transforms
from random import random
from PIL import Image, ImageFilter

import skimage

import numpy as np


ROTATION_ANGLES = [90, 120, 180]

# --------------------------------------- Rotations ----------------------------------------
class RotateRandom(object):
    """Rotate PIL Image by 0, 90, 180, 270, randomly"""
    def __call__(self, img):
        random_rotation = np.random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = img.rotate(random_rotation*90)
        return img
    
class RotateFixedAngle(object):
    """Rotate PIL Image a determined angle"""
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, image):
        return image.rotate(self.angle)

    def __repr__(self):
        return "custom augmentation"
    
# -------------------------------------- Flip Augmentations -------------------------------------------
class RandomVerticalFlip(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        if random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    
class RandomHorizontalFlip(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        if random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
# ------------------------------------ Noise Augmentations --------------------------------------------

class RandomHEStain(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """

    def __call__(self, img):
        img_he = skimage.color.rgb2hed(img)
        img_he[:, :, 0] = img_he[:, :, 0] * np.random.normal(1.0, 0.02, 1)  # H
        img_he[:, :, 1] = img_he[:, :, 1] * np.random.normal(1.0, 0.02, 1)  # E
        img_rgb = np.clip(skimage.color.hed2rgb(img_he), 0, 1)
        img = Image.fromarray(np.uint8(img_rgb*255.999))
        return img

class RandomGaussianNoise(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """

    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(np.random.normal(0.0, 0.5, 1)))
        return img
    

# ----------------------------------- Normalize Augmentations ----------------------------------------

# Transformations applied to every bag (normalization)
normal_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


aug_transforms = [
    transforms.Compose([
        RandomHEStain(),
        RandomGaussianNoise(),
        RotateRandom(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        normal_transforms
    ]),
    transforms.Compose([
        RandomHEStain(),
        RandomGaussianNoise(),
        RotateRandom(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        normal_transforms
    ])
]

