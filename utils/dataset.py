from glob import glob

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

#TODO: edit dataset
class Screen(Dataset):
    def __init__(self, path: str):
        self._filepaths = glob(path)

    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)

    def __len__(self):
        return len(self)