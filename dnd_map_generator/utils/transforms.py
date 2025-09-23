"""
File for holding image transforms

Author: Peter Thomas
Date: September 23, 2025
"""
import numpy as np


def normalize(img):
    normalized = np.zeros(img.shape, dtype=np.float32)
    for channel in (0, 1, 2):
        normalized[channel] = 2 * (img[channel] - img[channel].min()) / (img[channel].max() - img[channel].min()) - 1.
    return normalized


def standardize(img_tensor):
    """
    Standardize image tensor to have mean 0 and std 1
    Args:
        img_tensor: torch.Tensor of shape (C, H, W) with pixel values in [0, 1]
    Returns:
        standardized_img: torch.Tensor of shape (C, H, W) with mean 0 and std 1
    """
    mean = img_tensor.mean([1, 2])
    std = img_tensor.std([1, 2]) + 1e-7  # Add a small value to avoid division by zero
    standardized_img = (img_tensor - mean) / std
    return standardized_img, mean, std

def rewhiten_image(img_tensor, mean, std):
    """
    Rewhiten image tensor to original mean and std
    Args:
        img_tensor: torch.Tensor of shape (C, H, W) with mean 0 and std 1
        mean: torch.Tensor of shape (C,) original mean
        std: torch.Tensor of shape (C,) original std
    Returns:
        rewhitened_img: torch.Tensor of shape (C, H, W) with original mean and std
    """
    rewhitened_img = img_tensor * std[:, None, None] + mean[:, None, None]
    return rewhitened_img


class StandardizeTransform:
    def __call__(img_tensor):
        """
        Standardize image tensor to have mean 0 and std 1
        Args:
            img_tensor: torch.Tensor of shape (C, H, W) with pixel values in [0, 1]
        Returns:
            standardized_img: torch.Tensor of shape (C, H, W) with mean 0 and std 1
        """
        mean = img_tensor.mean([1, 2])
        std = img_tensor.std([1, 2]) + 1e-7  # Add a small value to avoid division by zero
        standardized_img = (img_tensor - mean) / std
        return standardized_img