import os
import numpy as np
from typing import Tuple
from PIL import Image as Img
from torch.utils.data import Dataset


class CarvanaSet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Tuple:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")
        )
        # to augment using Albumentation, load into numpy array.
        image = np.array(Img.open(img_path).convert("RGB"))
        mask = np.array(Img.open(mask_path).convert("L"), dtype=np.float32)
        # normalize/convert white value from 255 to 1.0
        mask[mask == 255] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return (image, mask)
