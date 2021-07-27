from glob import glob

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from typing import Tuple

class Screen(Dataset):
    def __init__(self, path: str):
        self.paths = glob(path)

    def __getitem__(
        self, 
        index:int, 
        roi:Tuple[int] = (200, 800, 250, 600)
        ) -> Tuple[torch.Tensor, str]:

        img_path, target = self.paths[index], self.paths[index].split('/')[-1]
        img = cv2.imread(img_path)
        y1, y2, x1, x2 = roi
        img = img[y1:y2, x1:x2]
        img = np.array(img)
        img = img / 255
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = torch.Tensor(img).reshape(3, 256, 256)

        return img, target
    def __len__(self) -> int:
        return len(self.paths)