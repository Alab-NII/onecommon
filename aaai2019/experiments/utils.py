# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Various helpers.
"""

import random
import copy
import pdb

import torch
import numpy as np


def backward_hook(grad):
    """Hook for backward pass."""
    print(grad)
    pdb.set_trace()
    return grad

def save_model(model, file_name):
    """Serializes model to a file."""
    if file_name != '':
        with open(file_name, 'wb') as f:
            torch.save(model, f)


def load_model(file_name):
    """Reads model from a file."""
    with open(file_name, 'rb') as f:
        return torch.load(f)


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
