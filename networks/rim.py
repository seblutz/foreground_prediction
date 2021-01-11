"""This file contains network definitions for the recurrent inference machine."""

__author__ = 'Sebastian Lutz'
__email__ = 'lutzs@scss.tcd.ie'
__copyright__ = 'Copyright 2020, Trinity College Dublin'


import torch
import torch.nn as nn

from networks.ops import SpectralNorm


class SpectralRIM(nn.Module):
    """Recurrent inference machine with spectral normalization."""

    def __init__(self, kernel_size=3, features=32, stride=2, sigma=1):
        """Initialize the module.

        Arguments:
            kernel_size: Kernel size of the convolutions.
            features: Number of channels for the intermediate feature maps.
            stride: Stride for the pooling operation.
            sigma: Standard deviation of the normal distribution that is needed for the calculation of the gradient of the log-likelihood."""

        super(SpectralRIM, self).__init__()
        self.sigma = sigma
        input_nc = 7  # RGB FG + RGB BG + Alpha

        padding = (kernel_size - 1) // 2  # Calculate padding based on the kernel size.
        # The pooling operation is a strided convolution.
        pool = lambda x, n: SpectralNorm(nn.Conv2d(x, n, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        # No normalization, but using tanh as activation.
        norm = lambda x: nn.Sequential(
                    nn.Tanh()
               )
        # The unpooling operation is a transposed convolution.
        unpool = lambda x, n: SpectralNorm(nn.ConvTranspose2d(x, n, kernel_size=3, stride=stride, padding=1, output_padding=1))

        # The rnn part of the network. Input -> pool -> norm -> ConvGRU -> unpool -> norm -> ConvGRU.
        # This means there is one ConvGRU at half spatial size and one at full spatial size.
        self.rnn = MultiRNN([EmbeddingWrapper(ConvGRU(features, 4*features), pool(2*input_nc, features), norm(features))]
                          + [EmbeddingWrapper(ConvGRU(features, 4*features), unpool(4*features, features), norm(features))])
        # Final convolution at the end to reduce the number of channels back to the number of input channels.
        self.out = nn.Conv2d(4*features, input_nc, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, y, x, h, grad_mask=None):
        """Forward pass of this module.

        Arguments:
            y: The observed image, a torch.Tensor [Bx3xHxW].
            x: The current candidate solution, [FG, BG, Alpha], a torch.Tensor [Bx7xHxW].
            h: A list of hidden states for the rnn cells.
            grad_mask: A binary mask that can optionally get applied to the gradient of the log-likelihood."""

        # Calculate the gradient of the log-likelihood of the image given the candidate solutions.
        grad = gradient(y, x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:, :, :], self.sigma)
        # Apply binary mask if applicable.
        if grad_mask is not None:
            grad = grad * grad_mask
        # Append the candidate solutions and the gradient together.
        combined = torch.cat([x, grad], dim=1)
        # Forward pass.
        out, h = self.rnn(combined, h)
        d_x = self.out(out)

        return x + d_x, h


class MultiRNN(nn.Module):
    """This is a multi rnn module that calls multiple rnn cells in sequence."""

    def __init__(self, cells):
        """Initialize the module.

        Arguments:
            cells: List of rnn cells that should be called in sequence."""

        super(MultiRNN, self).__init__()

        # Save the cells as individual modules.
        self.num_cells = len(cells)
        for i, cell in enumerate(cells):
            self.add_module('cell{}'.format(i), cell)

    def forward(self, x, states):
        """Forward pass of this module.

        Arguments:
            x: Input torch.Tensor.
            states: List of hidden states for the individual cell modules."""

        new_states = []  # Save the updated hidden states in a list.
        # Iterate through all inner cells and call them in sequence.
        for i in range(self.num_cells):
            cell = getattr(self, 'cell{}'.format(i))
            if states is not None:
                h = states[i]
            else:
                h = None
            x, h_next = cell(x, h)
            new_states.append(h_next)

        return x, new_states


class EmbeddingWrapper(nn.Module):
    """This is an embedding wrapper around a rnn cell module.
    Calls will always be in the order: Embedding(x) -> (Normalization(x)) -> Cell(x,h)."""

    def __init__(self, cell, embedding_func, normalizer=None):
        """Inititialize the module.

        Arguments:
            cell: The internal module.
            embedding_func: The embedding function that will always be called first on any input.
            normalizer: An optional normalization function."""

        super(EmbeddingWrapper, self).__init__()

        self.embedding_func = embedding_func
        self.cell = cell
        self.normalizer = normalizer

    def forward(self, x, h):
        """Forward pass of this module.

        Arguments:
            x: Input torch.Tensor.
            h: Hidden state for the cell module."""

        # Call the embedding function only on the input tensor.
        x = self.embedding_func(x)

        # Normalize if applicable.
        if self.normalizer is not None:
            x = self.normalizer(x)

        # Call the cell module with the embedded input and the hidden state.
        return self.cell(x, h)


class ConvGRU(nn.Module):
    """Convolutional GRU module."""

    def __init__(self, input_nc, hidden_nc, kernel_size=3):
        """Initialize the module.

        Arguments:
            input_nc: Number of input channels.
            hidden_nc: Number of hidden channels == number of output channels.
            kernel_size: Kernel size for the convolution."""

        super(ConvGRU, self).__init__()

        # Calculate padding from kernel size and create the needed convolutional layers.
        padding = (kernel_size - 1) // 2
        self.hidden_nc = hidden_nc
        self.reset = nn.Conv2d(input_nc + hidden_nc, hidden_nc, kernel_size=kernel_size, stride=1, padding=padding)
        self.update = nn.Conv2d(input_nc + hidden_nc, hidden_nc, kernel_size=kernel_size, stride=1, padding=padding)
        self.candidate = nn.Conv2d(input_nc + hidden_nc, hidden_nc, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x, h):
        """Forward pass of this module.

        Arguments:
            x: Input torch.Tensor.
            h: Hidden state of this module, which is the same as the previous output."""

        # Initialize the hidden state if there is none available.
        if h is None:
            h = self.init_hidden(x)

        # Append x and h together.
        combined = torch.cat([x, h], dim=1)

        # Use Conv2d for the reset and update gate.
        reset_gate = torch.sigmoid(self.reset(combined))
        update_gate = torch.sigmoid(self.update(combined))

        # Use Conv2d for the candidate neural memory.
        combined = torch.cat([x, reset_gate * h], dim=1)
        candidate = self.candidate(combined)
        candidate_neural_memory = torch.tanh(candidate)

        # Calculate the next hidden state.
        h_next = update_gate * h + (1. - update_gate) * candidate_neural_memory

        # Return the output in the form of [output, hidden_state] so this can be more easily exchanged with ConvLSTM.
        return h_next, h_next

    def init_hidden(self, x):
        """Initialize the hidden state of this module with zeros.

        Arguments:
            x: torch.Tensor that can serve as an example input for this module. We use this to get the spatial sizes we need."""

        b, _, h, w = x.shape
        h = torch.zeros(b, self.hidden_nc, h, w).to(x.device)

        return h


def gradient(image, foreground, background, alpha, sigma=1):
    """Calculate the gradient of the log-likelihood of the observed image given foreground/background/alpha.
    This corresponds to equation 5 in our paper.

    Arguments:
        image: The observed image, a torch.Tensor [Bx3xHxW].
        foreground: The current candidate solution for the foreground colors, a torch.Tensor [Bx3xHxW].
        background: The current candidate solution for the background colors, a torch.Tensor [Bx3xHxW].
        alpha: The current candidate solution for the alpha matte, a torch.Tensor [Bx1xHxW].
        sigma: Standard deviation of the normal distribution.

    Returns:
        The gradient of the log-likelihood of the observed image given foreground/background/alpha."""

    # The results will be a torch.Tensor with 7 channels, 3 for foreground/background respectively + 1 for alpha.
    # All channels share one static part of the calculation with an individual prefix.
    static = -2. * (image - (alpha * foreground - (1. - alpha) * background))
    foreground_gradient = static * -alpha
    background_gradient = static * (1. - alpha)
    alpha_gradient_step = static * (-foreground - background)
    alpha_gradient = torch.unsqueeze(alpha_gradient_step[:, 0, :, :] + alpha_gradient_step[:, 1, :, :] + alpha_gradient_step[:, 2, :, :], dim=1)

    return torch.cat([foreground_gradient, background_gradient, alpha_gradient], dim=1) / (sigma**2)
