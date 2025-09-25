"""
Utilities for plotting histograms and statistics for image datasets.

Author: Peter Thomas
Date: September 24, 2025
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def collect_images(root_image_dir):
    image_paths = []
    for path, folders, files in os.walk(root_image_dir):
        if os.path.basename(path) == "__MACOSX":
            continue
        for folder in folders:
            if folder == "__MACOSX":
                continue
            else:
                for elem in os.listdir(os.path.join(path, folder)):
                    if elem.endswith(('.png', '.jpg', '.jpeg')) and not elem.startswith("Promo"):
                        image_paths.append(os.path.join(path, folder, elem))
    return image_paths


def get_mean_std(image_paths):

    means = []
    stds = []

    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        np_image = np.array(image) 
        means.append(np.mean(np_image, axis=(0, 1)))
        stds.append(np.std(np_image, axis=(0, 1)))

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std


def plot_histograms(image_paths):

    all_pixels = { 'R': [], 'G': [], 'B': [] }

    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        np_image = np.array(image).astype(np.float32)
        all_pixels['R'].extend(np_image[:, :, 0].flatten())
        all_pixels['G'].extend(np_image[:, :, 1].flatten())
        all_pixels['B'].extend(np_image[:, :, 2].flatten())

    fig = plt.figure(figsize=(15, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.5)
    for ax, channel in zip(grid, ['R', 'G', 'B']):
        ax.hist(all_pixels[channel], bins=256, color=channel.lower(), alpha=0.7)
        ax.set_title(f'{channel} Channel Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_xlim([0, 255])
        ax.grid(True)

    return fig
#    plt.savefig(os.path.join(output_plot_dir, 'rgb_histograms.png'))
#        plt.figure()
#        plt.hist(all_pixels[channel], bins=256, color=channel.lower(), alpha=0.7)
#        plt.title(f'{channel} Channel Histogram')
#        plt.xlabel('Pixel Intensity')
#        plt.ylabel('Frequency')
#        plt.xlim([0, 255])
#        plt.grid(True)
#        plt.savefig(os.path.join(output_plot_dir, f'{channel}_histogram.png'))
#        plt.close()


def plot_differences_in_per_img_stats_and_per_dataset_stats(image_paths, mean, std):

    diff_mean_r = []
    diff_mean_g = []
    diff_mean_b = []

    diff_std_r = []
    diff_std_g = []
    diff_std_b = []

    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        np_image = np.array(image).astype(np.float32)
        mean_r = np_image[:, :, 0].mean()
        mean_g = np_image[:, :, 1].mean()
        mean_b = np_image[:, :, 2].mean()

        std_r = np_image[:, :, 0].std()
        std_g = np_image[:, :, 1].std()
        std_b = np_image[:, :, 2].std()

        diff_mean_r.append(mean_r - mean[0])
        diff_mean_g.append(mean_g - mean[1])
        diff_mean_b.append(mean_b - mean[2])

        diff_std_r.append(std_r - std[0])
        diff_std_g.append(std_g - std[1])
        diff_std_b.append(std_b - std[2])

    # Plotting differences in means and stds
    fig = plt.figure(figsize=(15, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.5)
    for ax, diffs, title in zip(grid, 
                                [diff_mean_r, diff_mean_g, diff_mean_b, diff_std_r, diff_std_g, diff_std_b],
                                ['Mean Difference R', 'Mean Difference G', 'Mean Difference B',
                                 'Std Dev Difference R', 'Std Dev Difference G', 'Std Dev Difference B']):
        ax.hist(diffs, bins=50, color='blue', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Difference')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot histograms for image dataset")
    parser.add_argument('--root_image_dir', type=str, required=True, help='Root directory of the image dataset')
    parser.add_argument('--output_plot_dir', type=str, required=True, help='Directory to save the output plots')
    args = parser.parse_args()

    image_paths = collect_images(args.root_image_dir)
    print(f"Found {len(image_paths)} images.")

    mean, std = get_mean_std(image_paths)
    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std: {std}")

    if not os.path.exists(args.output_plot_dir):
        os.makedirs(args.output_plot_dir)

    fig = plot_histograms(image_paths)
    fig.savefig(os.path.join(args.output_plot_dir, 'rgb_histograms.png'))
    plt.close(fig)
    print(f"Histograms saved to {args.output_plot_dir}.")

    fig = plot_differences_in_per_img_stats_and_per_dataset_stats(image_paths, mean, std)
    fig.savefig(os.path.join(args.output_plot_dir, 'mean_std_differences.png'))
    plt.close(fig)
    print(f"Mean and Std Dev differences saved to {args.output_plot_dir}.")