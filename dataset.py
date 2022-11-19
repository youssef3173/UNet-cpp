# Create a Class to Process the Cityscapes Dataset:

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from collections import namedtuple

from torchvision.transforms import InterpolationMode


class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', mode='fine', augment=False):

        self.root = os.path.expanduser(root)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.mode = 'gtFine'            # we don't have Coarse Dataset 
        self.images_dir = os.path.join( self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join( self.root, self.mode, split)
        self.split = split
        self.augment = augment
        self.images = []
        self.targets = []

        self.labels = [
            # name  id   trainId   category   catId     hasInstances   ignoreInEval   color
            ('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
            ('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
            ('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
            ('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
            ('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
            ('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
            ('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
            ('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
            ('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
            ('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),            
            ('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
            ('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
            ('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
            ('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
            ('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),            
            ('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
            ('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
            ('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
            ('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
            ('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),            
            ('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
            ('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
            ('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
            ('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
            ('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),            
            ('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
            ('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
            ('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
            ('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
            ('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),            
            ('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
            ('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
            ('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
            ('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
            ('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142))
        ]
  

        self.mapping = { label[1]:max( 0, label[4]) for label in self.labels }
        self.mappingrgb = { label[1]:label[7] for label in self.labels }

        self.num_classes = len( { self.mapping[key] for key in self.mapping})

        # =============================================
        # Check that inputs are valid
        # =============================================
        if mode == 'fine' and split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode "fine"! Please use split="train", split="test" or split="val"')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        # =============================================
        # Read in the paths to all images
        # =============================================
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], '{}_labelIds.png'.format(self.mode))
                # target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], '{}_color.png'.format(self.mode))
                self.targets.append(os.path.join(target_dir, target_name))

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Mode: {}\n'.format(self.mode)
        fmt_str += '    Augment: {}\n'.format(self.augment)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def __len__(self):
        return len(self.images)

    def mask_to_class(self, mask):
        '''
        Given the cityscapes dataset, this maps to a 0..classes numbers.
        This is because we are using a subset of all masks, so we have this "mapping" function.
        This mapping function is used to map all the standard ids into the smaller subset.
        '''
        maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            maskimg[mask == k] = self.mapping[k]
        return maskimg

    def mask_to_rgb(self, mask):
        '''
        Given the Cityscapes mask file, this converts the ids into rgb colors.
        This is needed as we are interested in a sub-set of labels, thus can't just use the
        standard color output provided by the dataset.
        '''
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def class_to_rgb(self, mask):
        '''
        This function maps the classification index ids into the rgb.
        For example after the argmax from the network, you want to find what class
        a given pixel belongs too. This does that but just changes the color
        so that we can compare it directly to the rgb groundtruth label.
        '''
        mask2class = dict((v, k) for k, v in self.mapping.items())
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mask2class:
            rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
            rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
            rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
        return rgbimg

    def __getitem__(self, index):
        # first load the RGB image
        image = Image.open(self.images[index]).convert('RGB')

        # next load the target
        target = Image.open(self.targets[index]).convert('L')

        # If augmenting, apply random transforms
        # Else we should just resize the image down to the correct size
        if self.augment:
            # Resize
            image = TF.resize(image, size=(128+10, 256+10), interpolation=InterpolationMode.BILINEAR)
            target = TF.resize(target, size=(128+10, 256+10), interpolation=InterpolationMode.NEAREST)
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(128, 256))
            image = TF.crop(image, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
            # Random vertical flipping
            # (I found this caused issues with the sky=road during prediction)
            # if random.random() > 0.5:
            #    image = TF.vflip(image)
            #    target = TF.vflip(target)
        else:
            # Resize
            image = TF.resize(image, size=(128, 256), interpolation=InterpolationMode.BILINEAR)
            target = TF.resize(target, size=(128, 256), interpolation=InterpolationMode.NEAREST)

        # convert to pytorch tensors
        # target = TF.to_tensor(target)
        target = torch.from_numpy(np.array(target, dtype=np.uint8))
        image = TF.to_tensor(image)

        # convert the labels into a mask
        targetrgb = self.mask_to_rgb(target)
        targetmask = self.mask_to_class(target)
        targetmask = targetmask.long()
        targetrgb = targetrgb.long()

        # finally return the image pair
        return image, targetmask, targetrgb
