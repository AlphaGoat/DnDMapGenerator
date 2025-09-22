"""
Dataset for training map generation diffusion model

Author: Peter Thomas
Date: September 21, 2025
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class DndMapDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def _load_data(self):
        data = []
        for elem in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, elem)):
                with open(os.path.join(self.data_dir, elem, "description.txt"), 'r') as f:
                    description = f.read().strip()
                filename = f"{"-".join([word.capitalize() for word in elem.split('_')])}"
                for subelem in os.listdir(os.path.join(self.data_dir, elem)):
                    if subelem.endswith(('.png', '.jpg', '.jpeg')) and filename in subelem:
                        data.append((os.path.join(self.data_dir, elem, subelem), description))
                
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, desc = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            "in": desc,
            "out": image
        }