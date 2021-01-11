"""This folder contains utility operations."""

__author__ = 'Sebastian Lutz'
__email__ = 'lutzs@scss.tcd.ie'
__copyright__ = 'Copyright 2020, Trinity College Dublin'


import os


def create_output_directories(save_dir):
    """Creates the output directory structure.

    Arguments:
        save_dir: Path to the directory in which the results will be saved."""

    for subdir in ['foreground', 'background', 'alpha', 'rgba']:
        new_dir = os.path.join(save_dir, subdir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)


def remove_prefix_state_dict(state_dict, prefix="module"):
    """Removes the prefix from the key of pretrained state dict.

    Arguments:
        state_dict: The state dict to be modified.
        prefix: The prefix to be removed."""

    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()

    return new_state_dict
