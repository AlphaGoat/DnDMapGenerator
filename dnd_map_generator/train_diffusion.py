import os
from tqdm import tqdm
from diffusers import UNet2DModel


def build_model():
    model = UNet2DModel(
        sample_size=64,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 256, 512),  # the number of output channels for each UNet block
        down_block_types=(  # the types of downsampling blocks used
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(  # the types of upsampling blocks used
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


def train(model,
          logger,
          train_dataloader, 
          optimizer,
          lr_scheduler, 
          num_epochs, 
          device):

    model.to(device)

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            # Move batch to device
            images = batch["out"].to(device)

            # Forward pass
            outputs = model(images).sample

            # Compute loss (mean squared error between input and output)
            loss = ((outputs - images) ** 2).mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)

            # Log every 100 steps
            if step % 100 == 0:
                logger.add_scalar("train/loss", loss.item(), epoch * len(train_dataloader) + step)
                

def main(args):
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train diffusion model for DnD map generation")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    args = parser.parse_args()

    main(args)