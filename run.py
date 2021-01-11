#!/usr/bin/env python

"""This file contains functions to run the method."""

__author__ = 'Sebastian Lutz'
__email__ = 'lutzs@scss.tcd.ie'
__copyright__ = 'Copyright 2020, Trinity College Dublin'


import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

import networks
import dataloader
import utils


class Engine(object):
    """Engine class that executes all the needed steps to do an inference run.
    - Building the GCA and RIM models.
    - Loading in the weights.
    - Creating the dataloader.
    - Iterating over all test images.
    - Saving the output."""

    def __init__(self, data_dir, save_dir, weight_dir, iterations, num_workers, tile_size, all_results, cpu, alpha_dir):
        """Initializes the engine.

        Arguments:
            data_dir: Path to the directory containing the image data.
            save_dir: Path to the directory in which the results will be saved.
            weight_dir: Path to the directory where the trained weights are stored.
            iterations: Number of RIM iterations.
            num_workers: Number of workers for the data loader.
            tile_size: Spatial size of the tile that should be processed from the image. Instead of the whole image,
                       [tile_size X tile_size] tiles will be processed.
            all_results: Boolean that is true if all intermediate results should be saved.
            cpu: Boolean that is true if inference should be done on CPU.
            alpha_dir: Optional path to a directory that contains alpha mattes that should be used instead of GCA."""

        self.iterations = iterations
        self.all_results = all_results
        self.save_dir = save_dir
        self.cpu = cpu
        self.alpha_dir = alpha_dir

        self._create_dataloader(data_dir, num_workers, alpha_dir)
        self._build_model()
        self._load_weights(weight_dir)

        utils.create_output_directories(save_dir)

        # Converter object from torch.Tensor to PIL image.
        self.to_pil = ToPILImage()

        # Normalization values.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        if not self.cpu:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        # Variables for tiled processing.
        self.tile_size = tile_size
        # self.rf = 11  # Receptive field.
        self.rf = 31  # Receptive field.
        self.scales = [0.5, 1]  # Scales of the hidden states. The first state only has half of the spatial size of the input image.

    def _create_dataloader(self, data_dir, num_workers, alpha_dir):
        """Creates the dataloader.

        Arguments:
            data_dir: Path to the directory containing the image data.
            num_workers: Number of workers for the data loader.
            alpha_dir: Optional path to a directory that contains alpha mattes that should be used instead of GCA."""

        dataset = dataloader.DataGenerator(data_dir, alpha_dir)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    def _build_model(self):
        """Building the model."""

        if not self.alpha_dir:
            self.gca = networks.get_generator(encoder='resnet_gca_encoder_29', decoder='res_gca_decoder_22')
            self.gca = self.gca.eval()
        self.rim = networks.SpectralRIM(sigma=0.1)
        self.rim = self.rim.eval()

        if not self.cpu:
            if not self.alpha_dir:
                self.gca.cuda()
            self.rim.cuda()

    def _load_weights(self, weight_dir):
        """Loading in the weights.

        Arguments:
            weight_dir: Path to the directory where the trained weights are stored."""

        if not self.alpha_dir:
            gca_path = os.path.join(weight_dir, 'gca.pth')
            checkpoint = torch.load(gca_path)
            self.gca.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
        rim_path = os.path.join(weight_dir, 'rim.pth')
        checkpoint = torch.load(rim_path)
        self.rim.load_state_dict(utils.remove_prefix_state_dict(checkpoint['rim_state_dict']), strict=True)

    def run(self):
        """Executing the inference."""

        with torch.no_grad():
            for image_dict in tqdm(self.dataloader):
                # Load in images.
                rgb, trimap, image_name, original_shape = image_dict['rgb'], image_dict['trimap'], image_dict['image_name'], image_dict['shape']

                if not self.cpu:
                    rgb = rgb.cuda()
                    trimap = trimap.cuda()

                assert rgb.shape[2:] == trimap.shape[2:], 'Error for image {}: RGB and Trimap images need to be of the same size.'.format(image_name)

                # Create foreground and background weights.
                fg_weight = trimap[:, 2:, :, :].float()
                bg_weight = trimap[:, 0:1:, :, :].float()

                # Predict alpha using GCA or use given alpha.
                if self.alpha_dir:
                    alpha = image_dict['alpha']
                    if not self.cpu:
                        alpha = alpha.cuda()
                else:
                    try:
                        alpha, _ = self.gca(rgb, trimap)
                    except Exception as e:
                        if 'out of memory' in str(e):
                            print('CUDA out of memory exception in image {}.'.format(image_name))
                            continue
                        elif 'DefaultCPUAllocator: can\'t allocate memory' in str(e):
                            print('CPU memory allocation error in image {}'.format(image_name))
                            continue
                        else:
                            sys.exc_info()
                            print('Error processing image {}'.format(image_name))
                            continue

                # Refine alpha using the trimap.
                trimap = trimap.argmax(dim=1, keepdim=True)
                alpha[trimap == 2] = 1
                alpha[trimap == 0] = 0

                # Create initial candidates. All values in range [-1, 1].
                rgb = rgb.mul_(self.std).add_(self.mean)
                rgb = (rgb * 2) - 1
                fg = rgb * fg_weight
                bg = rgb * bg_weight
                alpha = (alpha * 2) - 1
                x = torch.cat([fg, bg, alpha], dim=1)
                h = None

                # Iterate RIM.
                for i in range(self.iterations):
                    try:
                        if self.cpu:
                            x, h = self.rim(rgb, x, h)
                        else:
                            x, h = self._tiled_processing(rgb, x, h)
                        x = x.detach()
                        if self.all_results:
                            self._save_results(x, rgb, trimap, image_name, original_shape, i)
                    except Exception as e:
                        if 'out of memory' in str(e):
                            print('CUDA out of memory exception in image {}.'.format(image_name))
                            continue
                        elif 'DefaultCPUAllocator: can\'t allocate memory' in str(e):
                            print('CPU memory allocation error in image {}'.format(image_name))
                            continue
                        else:
                            sys.exc_info()
                            print('Error processing image {}'.format(image_name))
                            continue

                # Save final results.
                self._save_results(x, rgb, trimap, image_name, original_shape)

    def _save_results(self, x, rgb, trimap, name, shape, i=None):
        """Function to save the results. Assumes batch size of 1.

        Arguments:
            x: The candidate solution to be saved. A torch.Tensor of shape [Bx7xHxW].
            rgb: The input rgb image.
            trimap: The input trimap.
            name: The original image name.
            shape: The original shape of the input.
            i: Iteration number. If this is given, add the iteration number to the filename."""

        def normalize(t):
            """Normalizes torch.Tensor t from [-1, 1] to [0, 1]."""

            t = (t + 1) / 2
            t = torch.clamp(t, 0, 1)
            return t

        # Normalize images.
        h, w = shape
        rgb = normalize(rgb)

        alpha = x[:, 6:, :, :]
        alpha = normalize(alpha)
        alpha[trimap == 2] = 1
        alpha[trimap == 0] = 0
        alpha = alpha[0, :, :h, :w].cpu()
        alpha = self.to_pil(alpha)

        trimap = torch.cat([trimap, trimap, trimap], dim=1)  # Make sure shape of trimap matches shape of fg/bg.

        fg = x[:, :3, :, :]
        fg = normalize(fg)
        fg = torch.where(trimap == 2, rgb, fg)
        fg = torch.where(trimap == 0, torch.zeros_like(fg), fg)
        fg = fg[0, :, :h, :w].cpu()
        fg = self.to_pil(fg)

        bg = x[:, 3:6, :, :]
        bg = normalize(bg)
        bg = torch.where(trimap == 0, rgb, bg)
        bg = torch.where(trimap == 2, torch.zeros_like(bg), bg)
        bg = bg[0, :, :h, :w].cpu()
        bg = self.to_pil(bg)

        rgba = fg.copy()
        rgba.putalpha(alpha)

        # Save images.
        name = name[0] if i is None else '{}_{}{}'.format(name[0][:-4], i, name[0][-4:])
        alpha.save(os.path.join(self.save_dir, 'alpha', name))
        fg.save(os.path.join(self.save_dir, 'foreground', name))
        bg.save(os.path.join(self.save_dir, 'background', name))
        rgba.save(os.path.join(self.save_dir, 'rgba', name))

    def _tiled_processing(self, y, x, h):
        """Process an image in tiles.

        Arguments:
            y: The observed input RGB image. A torch.Tensor of shape [Bx3xHxW].
            x: The tensor containing the candidate solutions. A torch.Tensor of shape [Bx7xHxW].
            h: The list of the hidden states of the RIM.

        Returns:
            The new candidate solution.
            The list of new hidden states."""

        _, _, height, width = x.shape

        # Check if the image is smaller than the tile size.
        if height < self.tile_size and width < self.tile_size:
            x, h = self.rim(y, x, h)
            return x, h
        else:
            # The image needs to be tiled. Calculate number of vertical and horizontal tiles.
            vertical_tiles = int(np.ceil(height / (self.tile_size - self.rf + 1)))
            horizontal_tiles = int(np.ceil(width / (self.tile_size - self.rf + 1)))
            # Calculate padding.
            pad = (self.rf - 1) // 2
            bottom_pad = (vertical_tiles * (self.tile_size - self.rf + 1)) + pad - height
            left_pad = (horizontal_tiles * (self.tile_size - self.rf + 1)) + pad - width
            # Pad the input tensors.
            x = F.pad(x, (pad, left_pad, pad, bottom_pad))
            y = F.pad(y, (pad, left_pad, pad, bottom_pad))
            outputs = []  # Tiled results will be stored here.

            # Iterate over tiles and process.
            for i in range(vertical_tiles):
                for j in range(horizontal_tiles):
                    x_tiled = self._create_tensor_tile(x, i, j)
                    y_tiled = self._create_tensor_tile(y, i, j)
                    if h is not None:  # Create tiles of the hidden states.
                        h_tiled = self._create_hidden_tile(h, i, j)
                    else:
                        h_tiled = h

                    out = self.rim(y_tiled, x_tiled, h_tiled)
                    outputs.append(out)

            # If the initial hidden state was not yet defined, create a new empty list of hidden states.
            if h is None:
                sizes = [h_tiled.shape for h_tiled in outputs[0][1]]
                h = [torch.zeros(size[0], size[1], vertical_tiles * size[2], horizontal_tiles * size[3]).to(x.device) for size in sizes]

            # Put the tiled outputs back together to form a full sized output.
            k = 0
            for i in range(vertical_tiles):
                for j in range(horizontal_tiles):
                    out = outputs[k]
                    k += 1
                    x[:, :,
                      i*(self.tile_size - self.rf + 1)+pad:i*(self.tile_size - self.rf + 1)+self.tile_size-pad,
                      j*(self.tile_size - self.rf + 1)+pad:j*(self.tile_size - self.rf + 1)+self.tile_size-pad] = out[0][:, :, pad:-pad, pad:-pad]
                    for layer in range(len(h)):
                        h[layer][:, :,
                                 i*(int(self.tile_size * self.scales[layer])):i*(int(self.tile_size * self.scales[layer]))
                                                                              + int(self.tile_size * self.scales[layer]),
                                 j*(int(self.tile_size * self.scales[layer])):j*(int(self.tile_size * self.scales[layer]))
                                                                              + int(self.tile_size * self.scales[layer])] = out[1][layer]
            return x[:, :, pad:pad + height, pad:pad + width], h

    def _create_tensor_tile(self, x, i, j):
        """Create the [i,j] tile from a full size tensor.

        Arguments:
            x: The tensor to tile. A torch.Tensor of shape [BxCxHxW].
            i: Vertical index.
            j: Horizontal index.

        Returns:
            Tiled tensor."""

        x_tiled = x[:, :, i * (self.tile_size - self.rf + 1):i * (self.tile_size - self.rf + 1) + self.tile_size,
                    j * (self.tile_size - self.rf + 1):j * (self.tile_size - self.rf + 1) + self.tile_size]

        return x_tiled

    def _create_hidden_tile(self, h, i, j):
        """Create the [i,j] list of tiles from a full size list of hidden states.

        Arguments:
            h: The list of hidden states. Elements in the list are torch.Tensors of shape [BxCxHxW].
            i: Vertical index.
            j: Horizontal index.

        Returns:
            List of tiled hidden states."""

        h_tiled = [h__[:, :, i * (int(self.tile_size * scale)):i * (int(self.tile_size * scale)) + int(self.tile_size * scale),
                   j * (int(self.tile_size * scale)):j * (int(self.tile_size * scale)) + int(self.tile_size * scale)] for h__, scale in
                   zip(h, self.scales)]

        return h_tiled


if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser(description='Predict foreground/background colors and alpha matte from image.')
    parser.add_argument('data', type=str, help='Path to the directory containing the image data. Needs to contain the '
                                               'subdirectories \'rgb\' and \'trimap\' with matching image names.')
    parser.add_argument('save_dir', type=str, help='Path to the directory in which the results will be saved. '
                                                   'Will create \'foreground\', \'background\' and \'alpha\' subdirectories if they do not already '
                                                   'exist. The result files will have the same filename as the input files.')
    parser.add_argument('-t', type=int, default=5, help='Number of iterations in the recurrent inference machine. Default=5.')
    parser.add_argument('-w', type=str, default='weights', help='Path to the directory where the trained weights are stored. '
                                                                'Needs to contain a \'gca.pth\' and \'rim.pth\' file with the trained weights. '
                                                                '\n Default=\'weights\'.')
    parser.add_argument('-workers', type=int, default=4, help='Number of workers for the data loading. Default=4.')
    parser.add_argument('-tile_size', type=int, default=512, help='Spatial size of the tile that should be processed from the image. '
                                                                  'Instead of the whole image, [tile_size X tile_size] tiles will be processed.')
    parser.add_argument('-all', default=False, action='store_true', help='Set this option to save all intermediate results.')
    parser.add_argument('-cpu', default=False, action='store_true', help='Run the inference on CPU.')
    parser.add_argument('-alpha', default=None, help='Optional path to a directory containing alpha mattes, if an external matting algorithm was used.')

    args = parser.parse_args()

    # Create and run inference engine.
    engine = Engine(args.data, args.save_dir, args.w, args.t, args.workers, args.tile_size, args.all, args.cpu, args.alpha)
    engine.run()
