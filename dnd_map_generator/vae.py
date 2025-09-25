"""
Variational Autoencoder (VAE) model
https://arxiv.org/pdf/1312.6114

Author: Peter Thomas
Date: September 22, 2025
"""
import torch
import numpy as np
from collections import OrderedDict


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(3, 32, kernel_size=3, stride=(2, 2), padding=1)),
                    ("batch_norm1", torch.nn.BatchNorm2d(32)),
                    ("act1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1)),
                    ("batch_norm2", torch.nn.BatchNorm2d(64)),
                    ("act2", torch.nn.ReLU()),
                    ("flatten", torch.nn.Flatten()),
                    ("fc_latent", torch.nn.Linear(64 * 64 * 64, latent_dim + latent_dim)),
#                    ("fc_logvar", torch.nn.Linear(64 * 64 * 64, 256))
                ]
            )
        )
        self.decoder = torch.nn.Sequential(
            OrderedDict(
                [
                    ("fc", torch.nn.Linear(256, 64 * 64 * 64)),
                    ("unflatten", torch.nn.Unflatten(1, (64, 64, 64))),
                    ("deconv1", torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2, 2), padding=1, output_padding=1)),
                    ("batch_norm1", torch.nn.BatchNorm2d(32)),
                    ("act1", torch.nn.ReLU()),
                    ("deconv2", torch.nn.ConvTranspose2d(32, 3, kernel_size=3, stride=(2, 2), padding=1, output_padding=1)),
                    ("act2", torch.nn.Sigmoid()),
                ]
            )
        )

        self.kl_divergence = 0.0
        self.apply(weights_init)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples):
        z = torch.randn(num_samples, 256).to(next(self.parameters()).device)
        samples = self.decode(z)
        return samples

    def forward(self, x):
        mu, logvar = self.encode(x)
        self.kl_divergence = self.calc_kl_divergence(mu, logvar).sum()
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @classmethod
    def calc_kl_divergence(cls, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
#        return torch.mean(-0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1. - logvar)


def reconstruction_loss(recon_x, x):
    return torch.nn.functional.mse_loss(recon_x, x, reduction='mean')


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = torch.log(2 * np.pi)
    return torch.sum(
        -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi)
    )

def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
            module.bias.data.fill_(0)
        except AttributeError as e:
            print(f"Error initializing weights for {classname}: {e}")
            print("Skipping initialization")