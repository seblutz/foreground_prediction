"""This file contains network definitions for decoders."""

__author__ = 'Li Yaoyi'
__copyright__ = 'Copyright 2019, Li Yaoyi'
__licence__ = 'MIT'


from networks.decoders.resnet_dec import ResNet_D_Dec


class ResShortCut_D_Dec(ResNet_D_Dec):
    """Resnet decoder with shortcuts."""

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        """Initialize the module.

        Arguments:
            block: Basic resnet block to use.
            layers: List of number of layers to use in each stage.
            norm_layer: Normalization layer to use.
            large_kernel: Set to true if a large convolutional kernel should be used.
            late_downsample: Set to true if the first downsampling operation should be done one stage late."""

        super(ResShortCut_D_Dec, self).__init__(block, layers, norm_layer, large_kernel,
                                                late_downsample=late_downsample)

    def forward(self, x, mid_fea):
        """Forward pass of this module.

        Arguments:
            x: The input tensor, a torch.Tensor [BxCxHxW].
            mid_fea: A dictionary which contains intermediate feature maps for shortcut connections.

        Returns:
            out: Output tensor."""

        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x = self.layer3(x) + fea3
        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha, None
