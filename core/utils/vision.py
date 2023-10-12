import torch
import numpy as np
from torchvision.utils import make_grid, save_image

def save_grid(tensor_list, filename=None, nrow=10):
    grid = make_grid(tensor_list, nrow=nrow)
    if filename:
        save_image(grid, filename)
    return grid
