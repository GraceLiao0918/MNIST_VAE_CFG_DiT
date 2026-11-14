
from script_utils import *

from CFG_GMM import MLP,ConditionalVectorField,CFGVectorFieldODE,CFGTrainer

from MNIST_data import MNISTSampler

# Fourier Time Encoder
class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: b
        Returns:
        - embeddings: b d
        """
        # Step 1: compute frequencies f_i = 2 * pi * w_i * t
        # raise NotImplementedError("Fill me in!")

        # Step 2: compute sin(f_i) and cos(f_i)
        # raise NotImplementedError("Fill me in!")

        # Step 3: Concatenate and return
        # raise NotImplementedError("Fill me in!")

        # Step 1: compute frequencies f_i = 2 * pi * w_i * t
        frequencies = 2 * torch.pi * self.weights * t.unsqueeze(-1)

        # Step 2: compute sin(f_i) and cos(f_i)
        sin_features = torch.sin(frequencies)
        cos_features = torch.cos(frequencies)

        # Step 3: Concatenate and return
        return torch.cat([sin_features, cos_features], dim=-1)

# Patchifier
class Patchifier(nn.Module):
  def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int):
    super().__init__()
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"

    # raise NotImplementedError("Fill me in!")
    self.proj = nn.Conv2d(
    in_channels=c_in,
    out_channels=dim,
    kernel_size=patch_size,
    stride=patch_size,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: (bs, 1, img_size, img_size)
    Returns:
    - x: (bs, 1, img_size, img_size)
    """
    # raise NotImplementedError("Fill me in!")
    # x: (B, c_in, img_size, img_size)
    x = self.proj(x)          # (B, dim, H/p, W/p)
    B, D, Hp, Wp = x.shape
    x = x.flatten(2)          # (B, D, Hp*Wp)
    x = x.transpose(1, 2)     # (B, Hp*Wp, D)
    return x


# `MHA` (multi-headed self-attention), `DiffusionTransformerLayer`, and `DiffusionTransformer`.
class MHA(nn.Module):
  """
  Multi-headed self-attention
  """
  def __init__(self, dim: int, heads: int):
    super().__init__()
    assert dim % heads == 0

    # raise NotImplementedError("Fill me in!")
    self.heads = heads
    self.head_dim = dim // heads
    self.qkv = nn.Linear(dim, 3 * dim)
    self.proj = nn.Linear(dim, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    Returns:
    - x: b n d
    """
    b, n, d = x.shape
    # Compute queries, keys, and values
    qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    # raise NotImplementedError("Fill me in!")

    # Fold head into batch dimension
    q = q.reshape(b * self.heads, n, self.head_dim)
    k = k.reshape(b * self.heads, n, self.head_dim)
    v = v.reshape(b * self.heads, n, self.head_dim)
    # raise NotImplementedError("Fill me in!")

    # # Compute attention
    # raise NotImplementedError("Fill me in!")
    #
    #
    # # Combine with values
    # raise NotImplementedError("Fill me in!")
    #
    # # Unfold heads
    # raise NotImplementedError("Fill me in!")
    #
    # # Pass throuh FF and return
    # raise NotImplementedError("Fill me in!")
    #

    # Compute scaled dot-product attention.
    attn = torch.softmax(
        (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5),
        dim=-1,
    )

    # Combine attention weights with values.
    out = attn @ v

    # Unfold heads and combine their features.
    out = out.reshape(b, self.heads, n, self.head_dim)
    out = out.transpose(1, 2).reshape(b, n, d)

    return self.proj(out)

class DiffusionTransformerLayer(nn.Module):
  def __init__(
      self,
      dim: int,
      heads: int,
  ):
    """
    Args:
    - n_tokens: sequence length (for sake of positional embeddings)
    - dim: dimension of hidden layers
    - heads: number of attention heads
    """
    super().__init__()

    # # Init normalization
    # raise NotImplementedError("Fill me in!")
    #
    #
    # # Initialize conditioning to zero - stabilizes residual connection!
    # raise NotImplementedError("Fill me in!")
    #
    #
    # # Init attention
    # raise NotImplementedError("Fill me in!")
    #
    # # Init feedforward
    # raise NotImplementedError("Fill me in!")

      # AdaLN supplies the scale and shift, so LayerNorm needs no affine parameters.
    self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
    self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

    # Produce shift, scale, and gate values for both residual blocks.
    self.adaLN_modulation = nn.Sequential(
        nn.SiLU(),
        nn.Linear(dim, 6 * dim),
    )

    # Zero initialization makes the residual blocks initially pass x through.
    nn.init.zeros_(self.adaLN_modulation[-1].weight)
    nn.init.zeros_(self.adaLN_modulation[-1].bias)

    self.attn = MHA(dim, heads)
    self.ff = MLP([dim, 4 * dim, dim])

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    - c: b d
    Returns:
    - x: b n d
    """
    # # Compute conditioning gating, scaling, and bias
    # raise NotImplementedError("Fill me in!")
    #
    # # Attention + residual connection
    # raise NotImplementedError("Fill me in!")
    #
    # # Feedforward + residual connection
    # raise NotImplementedError("Fill me in!")
    shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = (
        self.adaLN_modulation(c).chunk(6, dim=-1)
    )

    # Modulate normalized tokens, then apply attention with a gated residual.
    x = x + gate_attn.unsqueeze(1) * self.attn(
        self.norm1(x) * (1 + scale_attn.unsqueeze(1))
        + shift_attn.unsqueeze(1)
    )

    # Apply the second modulated, gated residual block.
    x = x + gate_ff.unsqueeze(1) * self.ff(
        self.norm2(x) * (1 + scale_ff.unsqueeze(1))
        + shift_ff.unsqueeze(1)
    )

    return x

class DiffusionTransformer(nn.Module):
  def __init__(
      self,
      depth: int,
      n_tokens: int,
      dim: int,
      **layer_kwargs,
  ):
    """
    Args:
    - n_tokens: sequence length (for sake of positional embeddings)
    - dim: dimension of hidden layers
    - heads: number of attention heads
    - depth: number of layers
    """
    super().__init__()
    # raise NotImplementedError("Fill me in!")
    self.positional_embedding = nn.Parameter(torch.randn(1, n_tokens, dim))
    self.layers = nn.ModuleList([
        DiffusionTransformerLayer(dim=dim, **layer_kwargs)
        for _ in range(depth)
    ])


  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    - c: b d
    Returns:
    - x: b n d
    """
    # raise NotImplementedError("Fill me in!")
    x = x + self.positional_embedding
    for layer in self.layers:
      x = layer(x, c)
    return x

# Depatchifier
class Depatchifier(nn.Module):
  def __init__(self, img_size: int, patch_size: int, dim: int, final_dim: int, c_out: int):
    super().__init__()
    self.patch_size = patch_size
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"

    # raise NotImplementedError("Fill me in!")
    grid_size = img_size // patch_size
    self.norm = nn.LayerNorm(dim)
    self.mlp = MLP([dim, 4 * dim, final_dim * patch_size * patch_size])
    self.rearrange = Rearrange(
        "b (h w) (f p1 p2) -> b f (h p1) (w p2)",
        h=grid_size,
        w=grid_size,
        f=final_dim,
        p1=patch_size,
        p2=patch_size,
    )
    self.conv = nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    Returns:
    - x: b 1 32 32
    """
    # raise NotImplementedError("Fill me in!")
    x = self.norm(x)
    x = self.mlp(x)
    x = self.rearrange(x)
    return self.conv(x)

class DiffusionTransformerFlowModel(ConditionalVectorField):
  def __init__(
      self,
      img_size: int = 32,
      patch_size: int = 8,
      num_layers: int = 12,
      c: int = 1,
      dim: int = 256,
      heads: int = 4,
      final_dim: int = 10,
      n_classes: int = 11,
    ):
      super().__init__()
      # # 0. Construct time_embedder and y_embedder
      # self.time_embedder = None # Todo!
      # self.y_embedder = None # Todo!
      #
      # # 1. Construct patchifier
      # self.patchifier = None # Todo!
      #
      # # 2. Construct DiT
      # n_tokens = (img_size // patch_size) ** 2
      # self.dit = None # Todo!
      #
      # # 3. Construct de-patchifier
      # self.depatchifier = None # Todo!
      # Encode time to dim features; embed the class label to dim features.
      self.time_embedder = FourierEncoder(dim)
      self.y_embedder = nn.Embedding(n_classes, dim)

      # Convert image to a sequence of patch tokens.
      self.patchifier = Patchifier(
          img_size=img_size,
          patch_size=patch_size,
          c_in=c,
          dim=dim,
      )

      # Process patch tokens conditioned on time and class.
      n_tokens = (img_size // patch_size) ** 2
      self.dit = DiffusionTransformer(
          depth=num_layers,
          n_tokens=n_tokens,
          dim=dim,
          heads=heads,
      )

      # Convert patch tokens back into an image.
      self.depatchifier = Depatchifier(
          img_size=img_size,
          patch_size=patch_size,
          dim=dim,
          final_dim=final_dim,
          c_out=c,
      )

  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b 1 32 32
    - t: b 1 1 1
    - c: b 1 1 1
    Returns:
    - u_t^theta(x|y): b 1 32 32
    """
    # # 1. Embed time and y
    # raise NotImplementedError("Fill me in!")
    #
    #
    # # 2. Patchify
    # raise NotImplementedError("Fill me in!")
    #
    # # 3. Pass through DiT
    # raise NotImplementedError("Fill me in!")
    #
    # # 4. Depatchify
    # raise NotImplementedError("Fill me in!")
    #
    # return x
    # FourierEncoder expects t with shape (batch,).
    t = t.reshape(t.shape[0])
    c = self.time_embedder(t) + self.y_embedder(y)

    x = self.patchifier(x)
    x = self.dit(x, c)
    x = self.depatchifier(x)
    return x

##################
# Training utils #
##################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def visualize_output(model, path, samples_per_class: int = 10, num_timesteps: int = 100, guidance_scales: List[float] = [1.0, 3.0, 5.0], save_path: Optional[str] = None, use_tqdm: bool = True):


  # Graph
  fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

  for idx, w in enumerate(guidance_scales):
      # Setup ode and simulator
      ode = CFGVectorFieldODE(model, guidance_scale=w, null_label=10)
      simulator = EulerSimulator(ode)

      # Sample initial conditions
      y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
      num_samples = y.shape[0]
      x0 = path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)

      # Simulate
      ts = torch.linspace(0,0.999,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
      x1 = simulator.simulate(x0, ts, y=y, use_tqdm=use_tqdm)

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

class MNISTCFGTrainer(CFGTrainer):
  '''
  CFG Trainer with MNIST-specific callback
  '''
  def checkpoint(self, step: int):
    # Save model
    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_model.pt'))
    torch.save(self.opt.state_dict(), os.path.join(self.output_dir, f'step_{step:6d}_opt.pt'))

    # Save output visualization
    visualize_output(self.model, self.path, save_path=os.path.join(self.output_dir, f'step_{step:6d}_output.png'), use_tqdm=False)


# #################
# # Training code #
# #################
def train_dit_cfg():
    # Initialize our sampler
    sampler = MNISTSampler().to(device)

    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        # p_data = MNISTSampler(),
        p_data = sampler,
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Initialize model
    dit = DiffusionTransformerFlowModel(
        img_size = 32,
        patch_size = 4,
        num_layers = 8,
        c = 1,
        dim = 256,
        heads = 8,
        final_dim = 10,
        n_classes = 11,
    ).to(device)

    # Initialize trainer
    trainer = MNISTCFGTrainer(path = path, eta=0.35, null_label=10)

    # Train! You should have reasonable results in ~15 A100 minutes
    # losses, steps = trainer.train(model=dit, num_steps = 20000, lr=0.4e-3, batch_size=256, ckpt_every=1000)
    losses, steps = trainer.train(model=dit, num_steps = 5005, lr=0.4e-3, batch_size=64, ckpt_every=1000)

    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs. Step")
    # plt.show()
    plt.savefig('./result/result_total/DiT_loss_5000.png', dpi=120, bbox_inches="tight")
    plt.close()
    print("ok")

    # Play with these!
    samples_per_class = 10
    num_timesteps = 100
    guidance_scales = [1.0, 3.0, 5.0]

    visualize_output(
        model=dit,
        path=path,
        samples_per_class=samples_per_class,
        num_timesteps=num_timesteps,
        guidance_scales=guidance_scales,
    )
    # plt.show()
    # save_path=os.path.join('./result/result_total/DiT_5000/')#报错，得具体的文件名称

def main():
    train_dit_cfg()

if __name__ == '__main__':
    main()
