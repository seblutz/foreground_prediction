"""This file contains network definitions for GCA Matting [https://github.com/Yaoyi-Li/GCA-Matting]."""

__author__ = 'Li Yaoyi'
__copyright__ = 'Copyright 2019, Li Yaoyi'
__licence__ = 'MIT'


import torch
import torch.nn as nn

from networks import encoders, decoders


class Generator(nn.Module):
    """Generator module that builds a generator based on encoder and decoder description."""

    def __init__(self, encoder, decoder):
        """Initialize the module.

        Arguments:
            encoder: Name of the encoder part.
            decoder: Name of the decoder part."""

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, trimap):
        """Forward pass of this module.

        Arguments:
            image: The input image, a torch.Tensor [Bx3xHxW].
            trimap: The input trimap, a torch.Tensor [Bx1xHxW] or [Bx3xHxW].

        Returns:
            alpha: A torch.Tensor with the alpha prediction.
            info_dict: A dictionary filled with additional information."""

        inp = torch.cat((image, trimap), dim=1)
        embedding, mid_fea = self.encoder(inp)
        alpha, info_dict = self.decoder(embedding, mid_fea)

        return alpha, info_dict


def get_generator(encoder, decoder):
    """Returns a generator given an encoder and decoder description.

    Arguments:
        encoder: Name of the encoder part.
        decoder: Name of the decoder part.

    Returns:
        A Generator based on the encoder/decoder descriptions."""

    generator = Generator(encoder=encoder, decoder=decoder)
    return generator
