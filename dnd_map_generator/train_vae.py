"""
Train first stage VAE model for DnD map generation

Author: Peter Thomas
Date: September 22, 2025
"""
import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from vae import VariationalAutoencoder, reconstruction_loss


class VAEDataset(Dataset):
    def __init__(self, root_image_dir, transform):
        super(VAEDataset, self).__init__()
        self.root_image_dir = root_image_dir
        self.transform = transform

        # Collect images from root image directory
        self.image_paths = []
        for paths, folders, files in os.walk(self.root_image_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(paths, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        return image


def train(model, 
          dataset, 
          optimizer,
          logger,
          num_train_epochs):


    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for epoch in range(num_train_epochs):
        model.train()
        for i, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{num_train_epochs}", total=len(dataloader)):
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch)

            loss_mse = reconstruction_loss(batch, out)
            loss_kl = model.kl_divergence / (batch.size(0))

            loss = loss_mse + loss_kl
            loss.backward()
            optimizer.step()

            # Track losses
            logger.add_scalar("Loss/MSE", loss_mse.item(), epoch * len(dataloader) + i)
            logger.add_scalar("Loss/KL", loss_kl.item(), epoch * len(dataloader) + i)
            logger.add_scalar("Loss/Total", loss.item(), epoch * len(dataloader) + i)

            if i % 10 == 0:
                # Plot reconstructions
                out = out.detach().cpu()
                grid = torch.cat((batch, out), dim=0)
                grid = transforms.ToPILImage()(transforms.utils.make_grid(grid, nrow=batch.size(0)))
                logger.add_image("Reconstruction", transforms.ToTensor()(grid), epoch * len(dataloader) + i)


def main(args):

    # Form dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = VAEDataset(args.data_dir, transform)

    # Initialize model and optimizer
    model = VariationalAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize logger
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = SummaryWriter(log_dir=args.log_dir)

    # Train the model
    train(model, dataset, optimizer, logger, args.num_train_epochs)

    print("Training complete. Saving model...")
    model_save_path = os.path.join(args.log_dir, "vae_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VAE model for DnD map generation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs and model checkpoints.")
    args = parser.parse_args()
    main(args)