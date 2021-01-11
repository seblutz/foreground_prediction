"""This file contains network definitions for decoders."""

__author__ = 'Li Yaoyi'
__copyright__ = 'Copyright 2019, Li Yaoyi'
__licence__ = 'MIT'


from networks.decoders.resnet_dec import ResNet_D_Dec, BasicBlock
from networks.decoders.res_shortcut_dec import ResShortCut_D_Dec
from networks.decoders.res_gca_dec import ResGuidedCxtAtten_Dec


__all__ = ['res_shortcut_decoder_22', 'res_gca_decoder_22']


def _res_shortcut_D_dec(block, layers, **kwargs):
    """Constructs a resnet decoder with shortcut connections."""

    model = ResShortCut_D_Dec(block, layers, **kwargs)
    return model


def _res_gca_D_dec(block, layers, **kwargs):
    """Constructs a resnet decoder with guided contextual attention."""

    model = ResGuidedCxtAtten_Dec(block, layers, **kwargs)
    return model


def res_shortcut_decoder_22(**kwargs):
    """Constructs a resnet_encoder_14 model."""

    return _res_shortcut_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)


def res_gca_decoder_22(**kwargs):
    """Constructs a resnet_encoder_14 model."""

    return _res_gca_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)