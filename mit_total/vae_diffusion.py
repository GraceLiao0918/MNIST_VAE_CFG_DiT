import os
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import uuid
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import torch
import torch.nn as nn
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from einops import rearrange
from einops.layers.torch import Rearrange


from mnit_diffusion import MLP,MHA,Trainer,LabeledSampleable,MNISTSampler,CFGVectorFieldODE,EulerSimulator,GaussianConditionalProbabilityPath,DiffusionTransformerFlowModel,LinearAlpha,LinearBeta


# Residual Block 残差块
class ResidualBlock(nn.Module):
  """ Two applications of LN + convolution + non-linearity + residual connection """
  def __init__(self, channels: int, act: nn.Module = nn.SiLU):
    super().__init__()

    # Init norm, convolutions, and activations
    # raise NotImplementedError("Fill me in!")
    self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
    self.act = act()



    # Initialize the second convolution to zero - stabilizes training early on!
    nn.init.zeros_(self.conv2.weight)
    nn.init.zeros_(self.conv2.bias)

  def forward(self, x: torch.Tensor):
    # Res init
    x_skip = x

    # Norm
    # raise NotImplementedError("Fill me in!")
    x = self.norm(x)

    # First convolution
    # raise NotImplementedError("Fill me in!")
    x = self.conv1(x)

    # Activation
    # raise NotImplementedError("Fill me in!")
    x = self.act(x)

    # Second convolution
    # raise NotImplementedError("Fill me in!")
    x = self.conv2(x)

    # Return residual connection
    return x_skip + x

# Attention Block 注意力块
class AttnBlock(nn.Module):
  def __init__(self, channels: int):
    super().__init__()

    # Reshape
    self.reshape1 = Rearrange('b c h w -> b (h w) c')

    # Norm + attention
    # raise NotImplementedError("Fill me in!")
    self.norm1 = nn.LayerNorm(channels)
    self.attn = MHA(channels, heads=4)

    # Norm + ff
    # raise NotImplementedError("Fill me in!")
    self.norm2 = nn.LayerNorm(channels)
    self.ff = MLP([channels, 4 * channels, channels])

  def forward(self, x: torch.Tensor):
    b, c, h, w = x.shape
    x = self.reshape1(x)

    # Attention + residual connection
    # raise NotImplementedError("Fill me in!")
    x_skip = x
    x = x_skip + self.attn(self.norm1(x))

    # Feedforward + residual connection
    # raise NotImplementedError("Fill me in!")
    x_skip = x
    x = x_skip + self.ff(self.norm2(x))

    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class EncoderBlock(nn.Module):
  def __init__(self, in_channels: int, downsample_channels: Optional[int] = None):
    super().__init__()
    # raise NotImplementedError("Fill me in!")
    self.res1 = ResidualBlock(in_channels)
    self.res2 = ResidualBlock(in_channels)
    self.attn = AttnBlock(in_channels)

    if downsample_channels is not None:
      self.downsample = nn.Conv2d(
          in_channels=in_channels,
          out_channels=downsample_channels,
          kernel_size=3,
          stride=2,
          padding=1,
      )
    else:
      self.downsample = None

  def forward(self, x: torch.Tensor):
    # raise NotImplementedError("Fill me in!")
    x = self.res1(x)
    x = self.res2(x)
    x = self.attn(x)

    if self.downsample is not None:
      x = self.downsample(x)

    return x

class Encoder(nn.Module):
  def __init__(self, in_channels: int, hidden_channels: list[int]):
    super().__init__()

    # Initial conv2d
    self.init_conv = nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels[0], kernel_size=3, padding=1, stride=1)

    # Initialize channels
    ch_in = hidden_channels
    ch_out = hidden_channels[1:] + [None]
    blocks = []
    for in_c, out_c in zip(ch_in, ch_out):
      blocks.append(EncoderBlock(in_c, out_c))
    self.blocks = nn.ModuleList(blocks)

    # Predict z_mean 均值
    z_dim = hidden_channels[-1]
    self.z_mean = nn.Sequential(
      nn.GroupNorm(1, z_dim),
      nn.Conv2d(in_channels = z_dim, out_channels = z_dim, kernel_size=1, stride=1, padding=0),
    )

    # Scalar log-variance 方差
    self.logvar = nn.Parameter(torch.zeros(()))

  def forward(self, x: torch.Tensor):
    # raise NotImplementedError("Fill me in!")
    x = self.init_conv(x)

    for block in self.blocks:
        x = block(x)

    z_mean = self.z_mean(x)
    return z_mean, self.logvar



class DecoderBlock(nn.Module):
  def __init__(self, in_channels: int, upsample_channels: Optional[int] = None):
    super().__init__()
    self.res1 = ResidualBlock(in_channels)
    self.res2 = ResidualBlock(in_channels)
    self.attn = AttnBlock(in_channels)
    if upsample_channels is not None:
      self.upsample = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=in_channels, out_channels=upsample_channels, kernel_size=3, padding=1, stride=1),
      )
    else:
      self.upsample = None

  def forward(self, x: torch.Tensor):
    # raise NotImplementedError("Fill me in!")
    x = self.res1(x)
    x = self.res2(x)
    x = self.attn(x)

    if self.upsample is not None:
        x = self.upsample(x)

    return x

class Decoder(nn.Module):
  def __init__(self, out_channels: int, hidden_channels: list[int]):
    super().__init__()

    # Initialize channels
    ch_in = hidden_channels
    ch_out = hidden_channels[1:] + [None]
    blocks = []
    for in_c, out_c in zip(ch_in, ch_out):
      blocks.append(DecoderBlock(in_c, out_c))
    self.blocks = nn.ModuleList(blocks)

    # Predict mean
    x_dim = hidden_channels[-1]
    self.x_mean = nn.Sequential(
      nn.GroupNorm(1, x_dim),
      nn.Conv2d(in_channels = x_dim, out_channels = out_channels, kernel_size=1, stride=1, padding=0),
    )

    # Scalar log-variance
    self.logvar = nn.Parameter(torch.zeros(()))

  def forward(self, x: torch.Tensor):
    # raise NotImplementedError("Fill me in!")
    for block in self.blocks:
        x = block(x)

    x_mean = self.x_mean(x)
    return x_mean, self.logvar


class VAE(nn.Module):
  def __init__(self, data_channels: int, hidden_channels: list[int], beta: float = 0.1):
    super().__init__()
    self.beta = beta

    # Encoder
    self._encoder = Encoder(data_channels, hidden_channels)

    # Decoder
    self._decoder = Decoder(data_channels, list(reversed(hidden_channels)))

  def encode(self, x: torch.Tensor):
    return self._encoder(x)

  def decode(self, z: torch.Tensor):
    return self._decoder(z)

  def forward(self, x: torch.Tensor):
    z_mean, z_logvar = self.encode(x)
    z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
    x_mean, x_logvar = self.decode(z)
    return z_mean, z_logvar, x_mean, x_logvar

  def compute_loss(self, z_mean: torch.Tensor, z_logvar: torch.Tensor, x_mean: torch.Tensor, x_logvar: torch.Tensor, x_true: torch.Tensor):
    """ See display 85 from the text """
    # KL loss
    # raise NotImplementedError("Fill me in!")

    # Reconstruction loss
    # raise NotImplementedError("Fill me in!")

    # KL loss: KL(q(z|x) || N(0, I))
    kl_per_sample = 0.5 * (
    z_mean.square() + torch.exp(z_logvar) - 1 - z_logvar
        ).flatten(start_dim=1).sum(dim=1)
    kl_loss = self.beta * kl_per_sample.mean()

    # Reconstruction loss: Gaussian negative log-likelihood
    recon_per_element = 0.5 * (
    math.log(2 * math.pi)
    + x_logvar
    + (x_true - x_mean).square() * torch.exp(-x_logvar)
)
    recon_per_sample = recon_per_element.flatten(start_dim=1).sum(dim=1)
    recon_loss = recon_per_sample.mean()


    return kl_loss + recon_loss

class MNISTVAETrainer(Trainer):
  def __init__(self, mnist_sampleable: LabeledSampleable, batch_size: int = 64, **kwargs):
    super().__init__(**kwargs)
    self.mnist = mnist_sampleable
    self.batch_size = batch_size

  def get_train_loss(self):
    """ See display 85 from the text """
    x, y = self.mnist.sample(self.batch_size)
    z_mean, z_std, x_mean, x_std = self.model(x)
    return self.model.compute_loss(z_mean, z_std, x_mean, x_std, x)

  @torch.no_grad()
  def checkpoint(self, step: int):
    # Save model
    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:06d}_model.pt'))
    torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:06d}_opt.pt'))

    # Save output visualization, using x_mean as reconstruction
    b = 10
    x, _ = self.mnist.sample(b)
    _, _, x_mean, _ = self.model(x)
    x_all = torch.cat([x, x_mean], dim=0)
    grid = make_grid(x_all, nrow=b, normalize=True, value_range=(0,1))
    plt.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    plt.axis("off")
    plt.title("VAE Reconstruction")
    plt.savefig(os.path.join(self.output_dir, f'step_{step:06d}_output.png'))
    plt.close()

@torch.no_grad()
def visualize_latent_interpolation(x1: torch.Tensor, x2: torch.Tensor, vae: VAE, n_steps: int, save_path: Optional[str] = None):
   z1_mean, z1_logvar = vae.encode(x1)
   z1 = z1_mean + torch.exp(0.5 * z1_logvar) * torch.randn_like(z1_mean) # 1 c h w

   z2_mean, z2_logvar = vae.encode(x2)
   z2 = z2_mean + torch.exp(0.5 * z2_logvar) * torch.randn_like(z2_mean) # 1 c h w

   lambdas = torch.linspace(0, 1, n_steps).to(z1.device)
   zs = (1 - lambdas) * z1.unsqueeze(-1) + lambdas * z2.unsqueeze(-1) # 1 c h w n_steps
   zs = rearrange(zs, '1 c h w n -> n c h w')
   samples, _ = vae.decode(zs) # n_steps 1 h w

   grid = make_grid(samples, nrow=n_steps, normalize=True, value_range=(0, 1))
   plt.figure(figsize=(n_steps * 2, 2))
   plt.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
   plt.axis("off")
   plt.title("Latent Interpolation")
   if save_path is not None:
       plt.savefig(save_path)
       plt.close()
   else:
       plt.show()
   return samples

# # Create dataset + VAE
device = torch.device('cuda')
mnist = MNISTSampler().to(device)
vae = VAE(
   data_channels = 1,
   hidden_channels = [16, 32, 64, 128],
   beta = 10.0,
).to(device)
#
# Create Trainer + train
trainer = MNISTVAETrainer(
    mnist_sampleable = mnist,
    batch_size = 64,
)
# losses, steps = trainer.train(
#    model = vae,
#    num_steps = 5000,
#    lr = 1e-3,
#    warmup_steps = 500,
#    ckpt_every = 250,
# )

losses, steps = trainer.train(
   model = vae,
   num_steps = 1000,
   lr = 1e-3,
   warmup_steps = 500,
   ckpt_every = 250,
)

plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()

# Perform interpolation in the latent space
vae.eval()
# samples, _ = mnist.sample(2)

samples, labels = mnist.sample(2)
print(f"这次抽到的真实数字是：{labels[0]} 和 {labels[1]}")

interpolated_samples = visualize_latent_interpolation(
   x1 = samples[:1],
   x2 = samples[1:2],
   vae = vae,
   n_steps = 10,
) # n_steps 1 h w




class LatentCFGTrainer(Trainer):
    def __init__(
        self,
        mnist: MNISTSampler,
        vae: VAE,
        path: GaussianConditionalProbabilityPath,
        eta: float,
        null_label: int,
        eps: float = 0.001,
        **kwargs
    ):
        assert eta > 0 and eta < 1
        super().__init__(**kwargs)
        self.mnist = mnist
        self.vae = vae
        self.path = path
        self.eta = eta
        self.eps = eps
        self.path = path
        self.null_label = null_label

    @torch.no_grad()
    def visualize_samples(self, save_path: str, samples_per_class: int = 10, num_timesteps: int = 100, guidance_scales: List[float] = [1.0, 3.0, 5.0], use_tqdm = False):
      # Graph
      fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

      for idx, w in enumerate(guidance_scales):
          # Setup ode and simulator
          ode = CFGVectorFieldODE(self.model, guidance_scale=w, null_label=10)
          simulator = EulerSimulator(ode)

          # Sample initial conditions
          y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
          num_samples = y.shape[0]
          z0 = self.path.p_simple.sample(num_samples)

          # Simulate
          ts = torch.linspace(0,0.999,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
          z1 = simulator.simulate(z0, ts, y=y, use_tqdm=use_tqdm)

          # Decode
          x1, _ = self.vae.decode(z1)

          # Plot
          v_min, v_max = x1.min(), x1.max()
          x1 = (x1 - v_min) / (v_max - v_min)
          grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(0,1))
          axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
          axes[idx].axis("off")
          axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)

      # Save
      if save_path is not None:
          plt.savefig(save_path)
          plt.close()
      else:
        plt.show()

    plt.show()

    def checkpoint(self, step: int):
      # Save model
      torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_model.pt'))
      torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_opt.pt'))

      # Save output visualization
      self.visualize_samples(save_path=os.path.join(self.output_dir, f'step_{step:6d}_output.png'))

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z, y from MNIST + encode
        with torch.no_grad():
          # raise NotImplementedError("Fill me in!")
          x, y = self.mnist.sample(batch_size)
          z_mean, z_logvar = self.vae.encode(x)
          z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        # raise NotImplementedError("Fill me in!")
        mask = torch.rand(batch_size, device=y.device) < self.eta
        y = torch.where(mask, torch.full_like(y, self.null_label), y)

        # Step 3: Sample t and x
        # raise NotImplementedError("Fill me in!")
        t = torch.rand(batch_size, device=z.device, dtype=z.dtype) * (1 - self.eps)
        x_t = self.path.sample_conditional_path(z, t)

        # Step 4: Regress and output loss
        # raise NotImplementedError("Fill me in!")
        pred = self.model(x_t, t, y)
        target = self.path.conditional_vector_field(x_t, z, t)
        return ((pred - target).reshape(batch_size, -1) ** 2).sum(dim=1).mean()

# Finally, let's train!

vae = vae.to(device) # VAE(data_channels = 1, hidden_channels = [16, 32, 64, 128], beta = 1.0)

# Initialize latent probability path
c = 128
img_size = 4

path = GaussianConditionalProbabilityPath(
    p_data = None,
    p_simple_shape = [c, img_size, img_size],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Initialize model
dit = DiffusionTransformerFlowModel(
    img_size = img_size,
    patch_size = 1,
    num_layers = 8,
    c = c,
    dim = 256,
    heads = 8,
    final_dim = 10,
    n_classes = 11,
).to(device)

# Dataset
mnist = MNISTSampler().to(device)

# Initialize trainer
trainer = LatentCFGTrainer(
    mnist = mnist,
    vae = vae,
    path = path,
    eta=0.35,
    null_label=10
)

# Train! You should have reasonable results in ~15 A100 minutes
# losses, steps = trainer.train(model=dit, num_steps = 10000, lr=0.4e-3, batch_size=256, ckpt_every=500)

losses, steps = trainer.train(model=dit, num_steps = 2000, lr=0.1e-3, batch_size=64, ckpt_every=500)

plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss vs. Step")
plt.show()

