"""
Train first stage VAE model for DnD map generation

Author: Peter Thomas
Date: September 22, 2025
"""
import os
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .vae import VariationalAutoencoder, reconstruction_loss
from .utils.transforms import standardize, rewhiten_image
#from .utils.transform import Standardize
#from .utils.debug import print_sizes


class VAEDataset(Dataset):
    def __init__(self, root_image_dir, transform):
        super(VAEDataset, self).__init__()
        self.root_image_dir = root_image_dir
        self.transform = transform

        # Collect images from root image directory
        self.image_paths = []
        for path, folders, files in os.walk(self.root_image_dir):
            if os.path.basename(path) == "__MACOSX":
                continue
            for folder in folders:
                if folder == "__MACOSX":
                    continue
                else:
                    for elem in os.listdir(os.path.join(path, folder)):
                        if elem.endswith(('.png', '.jpg', '.jpeg')) and not elem.startswith("Promo"):
                            self.image_paths.append(os.path.join(path, folder, elem))

        print(f"Found {len(self.image_paths)} images for VAE training.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        return image


def train(model, 
          device,
          dataset, 
          optimizer,
          logger,
          num_train_epochs,
          batch_size):


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(num_train_epochs):
        model.train()
        for i, batch in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{num_train_epochs}", total=len(dataloader)):

            # Preprocess batch
            batch, mean, std = standardize(batch)
            batch = batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass
#            print_sizes(model.encoder, batch) if (epoch == 0 and i == 0) else None
            out, mu, logvar = model(batch)

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
                # Calculate and log gradient norms
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                logger.add_scalar("Gradients/Total_Norm", total_norm, epoch * len(dataloader) + i)

                # Log latent space statistics
                logger.add_scalar("Latent/Mu_Mean", mu.mean().item(), epoch * len(dataloader) + i)
                logger.add_scalar("Latent/Mu_Std", mu.std().item(), epoch * len(dataloader) + i)
                logger.add_scalar("Latent/LogVar_Mean", logvar.mean().item(), epoch * len(dataloader) + i)
                logger.add_scalar("Latent/LogVar_Std", logvar.std().item(), epoch * len(dataloader) + i)

                # Plot reconstructions
                out = out.detach().cpu()
                batch = batch.detach().cpu()


                grid = torch.cat((batch, out), dim=0)
                grid = transforms.ToPILImage()(torchvision.utils.make_grid(grid, nrow=batch.size(0)))
                logger.add_image("Reconstruction", transforms.ToTensor()(grid), epoch * len(dataloader) + i)


def main(args):

    # Form dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
#        Standardize(),
    ])
    
    dataset = VAEDataset(args.data_dir, transform)

    # Initialize model and optimizer
    model = VariationalAutoencoder()
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize logger
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = SummaryWriter(log_dir=args.log_dir)

    # Train the model
    train(model, args.device, dataset, optimizer, logger, args.num_train_epochs, args.batch_size)

    print("Training complete. Saving model...")
    model_save_path = os.path.join(args.log_dir, "vae_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VAE model for DnD map generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda or cpu).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--latent_dim", type=int, default=256, help="Dimensionality of the latent space.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs and model checkpoints.")
    args = parser.parse_args()

    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")

    main(args)