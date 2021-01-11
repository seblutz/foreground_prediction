"""This file contains classes and methods to read in image data and to convert it to torch.Tensors that can be ingested by the network."""

__author__ = 'Sebastian Lutz'
__email__ = 'lutzs@scss.tcd.ie'
__copyright__ = 'Copyright 2020, Trinity College Dublin'

import cv2
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms


# Definition of usable image extensions.
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP',
]


def is_image_file(filename):
    """Checks if the file is an image file.

    Arguments:
        filename: The filename of the file to be checked.

    Returns:
        True if the file is an image file, false otherwise."""

    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root_dir, full=True):
    """Lists all image files in a given directory.

    Arguments:
        root_dir: The root directory.
        full: Optional. True if the full path should be saved, False for only the filename.

    Returns:
        images: List of all image files in this directory and all subdirectories."""

    images = []
    assert os.path.isdir(root_dir), '%s is not a valid directory' % root_dir

    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in fnames:
            if is_image_file(fname):
                if full:
                    path = os.path.join(root, fname)
                else:
                    path = fname
                images.append(path)

    return images


class Padding(object):
    """Makes sure that the given images fit into the network and pads them if necessary.
    Since the network downscales and upscales feature maps, the spatial sizes of the input images need to be evenly
    divisible by 2^x, where x is the amount of downscaling and upscaling steps."""

    def __init__(self, steps):
        """Initializes the amount of upscaling and downscaling steps.

        Arguments:
            steps: Amount of upscaling and downscaling steps."""

        self.factor = 2**steps

    def __call__(self, sample):
        """Call method, pads the 'rgb' and 'trimap' arrays in the sample dictionary if necessary.

        Arguments:
            sample: Dictionary with 'rgb' and 'trimap' arrays.

        Returns:
            sample: Dictionary with 'rgb' and 'trimap' arrays that fit the network."""

        # Check the spatial size of the input images.
        h, w = sample['shape']

        # Check if the images need padding.
        if h % self.factor == 0 and w % self.factor == 0:
            return sample

        # Calculate padding amount.
        target_h = self.factor * ((h - 1) // self.factor + 1)
        target_w = self.factor * ((w - 1) // self.factor + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        # Pad images.
        padded_rgb = np.pad(sample['rgb'], ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((0, pad_h), (0, pad_w)), mode="reflect")

        sample['rgb'] = padded_rgb
        sample['trimap'] = padded_trimap
        if 'alpha' in sample.keys():
            padded_alpha = np.pad(sample['alpha'], ((0, pad_h), (0, pad_w)), mode="reflect")
            sample['alpha'] = padded_alpha

        return sample


class ToTensor(object):
    """Converts numpy.ndarrays to torch.Tensors and normalizes them appropriately."""

    def __init__(self):
        """Initialize the normalization values."""

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, sample):
        """Call method, converts the numpy.ndarrays in sample to torch.Tensors and normalizes them.

        Arguments:
            sample: Dictionary with 'rgb' and 'trimap' arrays.

        Returns:
            sample: Dictionary with converted and normalized entries for 'rgb' and 'trimap'."""

        # Convert from BGR (cv2) to RGB.
        rgb, trimap = sample['rgb'][:, :, ::-1], sample['trimap']

        # Swap axes from [HxWxC] to [CxHxW].
        rgb = rgb.transpose((2, 0, 1)).astype(np.float32)

        # Scale trimap values from [0, 255] to [0, 1, 2] for background, unknown, foreground.
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1

        # Normalize rgb image and convert to torch.Tensor.
        rgb /= 255.
        sample['rgb'], sample['trimap'] = torch.from_numpy(rgb), torch.from_numpy(trimap).to(torch.long)
        sample['rgb'] = sample['rgb'].sub_(self.mean).div_(self.std)
        sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()

        # Transform alpha if it was given.
        if 'alpha' in sample.keys():
            alpha = sample['alpha'] / 255.
            sample['alpha'] = torch.from_numpy(alpha).unsqueeze(0).float()

        return sample


class DataGenerator(Dataset):
    """Data generation class. Given a data directory that includes an 'rgb' and 'trimap' folder with corresponding rgb and trimap images,
    this class loads in the images and returns them as normalized torch.Tensors."""

    def __init__(self, data_dir, alpha_dir):
        """Initialize the generator.

        Arguments:
            data_dir: Path to a directory containing a 'rgb' and 'trimap' folder.
            alpha_dir: Optional path to a directory that contains alpha mattes that should be used instead of GCA."""

        self.rgb = sorted(make_dataset(os.path.join(data_dir, 'rgb')))
        self.trimap = sorted(make_dataset(os.path.join(data_dir, 'trimap')))
        if alpha_dir:
            self.alpha = sorted(make_dataset(alpha_dir))
        else:
            self.alpha = alpha_dir

        self.transform = transforms.Compose([Padding(5), ToTensor()])

    def __len__(self):
        """Returns the size of the dataset."""

        return len(self.rgb)

    def __getitem__(self, idx):
        """Getitem function to get the image/trimap pair for a given index.

        Arguments:
            idx: Index for the image/trimap pair.

        Returns:
            Dictionary containing the rgb and trimap images as normalized torch.Tensor, the image name and the original image shape."""

        # Read in images.
        rgb = cv2.imread(self.rgb[idx])
        trimap = cv2.imread(self.trimap[idx], 0)
        image_name = os.path.split(self.rgb[idx])[-1]

        # Create dictionary.
        sample = {'rgb': rgb, 'trimap': trimap, 'image_name': image_name, 'shape': rgb.shape[:2]}

        # Load in alpha mattes if they were given.
        if self.alpha:
            alpha = cv2.imread(self.alpha[idx], 0)
            if rgb.shape[:2] != alpha.shape[:2]:
                alpha = cv2.resize(alpha, rgb.shape[:2][::-1])
            sample['alpha'] = alpha

        # Pad the images spatially if necessary, convert them to torch.Tensors and normalize them.
        sample = self.transform(sample)

        return sample
