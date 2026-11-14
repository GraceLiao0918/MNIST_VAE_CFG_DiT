
from script_utils import *

from MNIST_data import MNISTSampler

from VAE import VAE

from DiT import DiffusionTransformerFlowModel

from CFG_GMM import CFGVectorFieldODE



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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_cfg_vae_dit():
    # Initialize latent probability path
    c = 128
    img_size = 4

    # vae = vae.to(device) # VAE(data_channels = 1, hidden_channels = [16, 32, 64, 128], beta = 1.0)

    vae = VAE(
        data_channels=1,
        hidden_channels=[16, 32, 64, 128],
        beta=10.0,
    ).to(device)

    # 加载权重 训练5000步----Problem
    vae_ckpt_path = "runs/delicate-orange-f3910e8f/step_005000_model.pt"
    vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
    # 冻结 VAE
    vae.eval()
    vae.requires_grad_(False)

    path = GaussianConditionalProbabilityPath(
        p_data=None,
        p_simple_shape=[c, img_size, img_size],
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)

    # Initialize model
    dit = DiffusionTransformerFlowModel(
        img_size=img_size,
        patch_size=1,
        num_layers=8,
        c=c,
        dim=256,
        heads=8,
        final_dim=10,
        n_classes=11,
    ).to(device)

    # Dataset
    mnist = MNISTSampler().to(device)

    # Initialize trainer
    trainer = LatentCFGTrainer(
        mnist=mnist,
        vae=vae,
        path=path,
        eta=0.35,
        null_label=10
    )

    # Train! You should have reasonable results in ~15 A100 minutes
    # losses, steps = trainer.train(model=dit, num_steps = 10000, lr=0.4e-3, batch_size=256, ckpt_every=500)
    # 训练600步 - ---Problem
    losses, steps = trainer.train(model=dit, num_steps=5005, lr=0.4e-3, batch_size=64, ckpt_every=500)

    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs. Step")
    # plt.show()
    plt.savefig('./result/result_total/CFG_VAE_DiT_loss_5000.png', dpi=120, bbox_inches="tight")
    plt.close()
    print("ok")


def main():
    train_cfg_vae_dit()

if __name__ == "__main__":
    main()







