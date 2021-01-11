"""This file contains network definitions for encoders."""

__author__ = 'Li Yaoyi'
__copyright__ = 'Copyright 2019, Li Yaoyi'
__licence__ = 'MIT'


from networks.encoders.resnet_enc import ResNet_D, BasicBlock
from networks.encoders.res_shortcut_enc import ResShortCut_D
from networks.encoders.res_gca_enc import ResGuidedCxtAtten


__all__ = ['res_shortcut_encoder_29', 'resnet_gca_encoder_29']


def _res_shortcut_D(block, layers, **kwargs):
    """Constructs a resnet encoder with shortcut connections."""

    model = ResShortCut_D(block, layers, **kwargs)
    return model


def _res_gca_D(block, layers, **kwargs):
    """Constructs a resnet encoder with guided contextual attention."""

    model = ResGuidedCxtAtten(block, layers, **kwargs)
    return model


def resnet_gca_encoder_29(**kwargs):
    """Constructs a resnet_encoder_29 model."""

    return _res_gca_D(BasicBlock, [3, 4, 4, 2], **kwargs)


def res_shortcut_encoder_29(**kwargs):
    """Constructs a resnet_encoder_25 model."""

    return _res_shortcut_D(BasicBlock, [3, 4, 4, 2], **kwargs)
