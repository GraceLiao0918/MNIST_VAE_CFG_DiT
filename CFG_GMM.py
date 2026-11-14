# combining `CFGTrainer`, `CFGVectorFieldODE`, and `MLPConditionalVectorField` to sample from a Gaussian mixture (`GMM`) instance

from script_utils import *



class ConditionalVectorField(nn.Module, ABC):
    """
    Conditional vector field u_t^theta(x|y)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: b ...
        - t: b
        - y: b
        Returns:
        - u_t^theta(x|y): b ...
        """
        pass

# CFGVectorFieldODE
class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, null_label: int, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale
        self.null_label = null_label

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b ...
        - t: b
        - y: b
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * self.null_label
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

# CFGTrainer
class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, eta: float, null_label: int, eps: float = 0.001, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(**kwargs)
        self.eta = eta
        self.eps = eps
        self.path = path
        self.null_label = null_label

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        # raise NotImplementedError("Fill me in!")
        z, y = self.path.p_data.sample(batch_size)

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        # raise NotImplementedError("Fill me in!")
        mask = torch.rand(batch_size, device=y.device) < self.eta
        y = torch.where(mask, torch.full_like(y, self.null_label), y)

        # Step 3: Sample t and x
        # raise NotImplementedError("Fill me in!")
        t = torch.rand(batch_size, device=z.device, dtype=z.dtype) * (1 - self.eps)
        x = self.path.sample_conditional_path(z, t)

        # Step 4: Regress and output loss
        # raise NotImplementedError("Fill me in!")
        pred = self.model(x, t, y)
        target = self.path.conditional_vector_field(x, z, t)
        return ((pred - target).reshape(batch_size, -1) ** 2).sum(dim=1).mean()


# Sanity check: implement MLPConditionalVectorField
class MLP(nn.Module):
  def __init__(self, dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU, final_init: bool = False):
    super().__init__()
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    self.net = torch.nn.Sequential(*mlp)

    if final_init:
      nn.init.zeros_(self.net[-1].weight)
      nn.init.zeros_(self.net[-1].bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    Returns:
    - x: b n d
    """
    return self.net(x)

class MLPConditionalVectorField(ConditionalVectorField):
  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      class_dim: int,
      num_classes: int
    ):
    super().__init__()
    self.mlp = MLP([dim + class_dim + 1, hidden_dim, hidden_dim, dim])
    self.class_embedding = nn.Embedding(num_classes + 1, class_dim)

  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
      """
      Args:
      - x: b d
      - t: b
      - y: b
      Returns:
      - u_t^theta(x|y): (b, c, h, w)
      """
      # raise NotImplementedError("Fill me in!")
      # 保存原始shape
      b, *rest, d = x.shape
      x_flat = x.reshape(b, d)   # 压成 [B,2]
      y_embed = self.class_embedding(y)
      t = t.unsqueeze(-1)
      inputs = torch.cat([x, y_embed, t], dim=-1)
      return self.mlp(inputs)


#####################
# Visualize Results #
#####################
def visualize_cfg_gmm(gmm,vector_field,path,device):
    # User knobs
    guidance_strength = 3.0  # try changing me!

    fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 6))

    # Panel 1: Target
    ax = axes[0]
    x_data, _ = gmm.sample(250)
    x_data = x_data.detach().cpu().numpy()
    ax.scatter(x_data[:, 0], x_data[:, 1], s=5, marker="*")
    ax.set_title("Target")

    # Panel 2: Conditioned on each mode
    ax = axes[1]
    cfg_vector_field = CFGVectorFieldODE(vector_field, guidance_scale=guidance_strength, null_label=3)
    simulator = EulerSimulator(cfg_vector_field)

    batch_size = 250
    labels = torch.arange(3).repeat_interleave(batch_size).to(device)
    x_init = path.p_simple.sample(3 * batch_size)  # b 2
    ts = torch.linspace(0, 1, 100).expand(3 * batch_size, -1).to(device)  # b nt
    xs = simulator.simulate(x_init, ts, y=labels)  # b 2
    for idx in range(3):
        xs_idx = xs[idx * batch_size: (idx + 1) * batch_size].detach().cpu().numpy()
        ax.scatter(xs_idx[:, 0], xs_idx[:, 1], s=5, label=f"Mode {idx}", marker="*")
    ax.legend()
    ax.set_title(f"CFG w/ Guidance Strength {guidance_strength:.2f}")

    # Panel 3: Unconditioned
    ax = axes[2]
    batch_size = 750
    labels = torch.ones(batch_size).long().to(device) * 3
    x_init = path.p_simple.sample(batch_size)  # b 2
    ts = torch.linspace(0, 1, 100).expand(batch_size, -1).to(device)  # b nt
    xs = simulator.simulate(x_init, ts, y=labels).detach().cpu().numpy()  # b 2
    ax.scatter(xs[:, 0], xs[:, 1], s=5, label=f"Mode {idx}", marker="*")
    ax.set_title(f"Unguided Samples")
    # plt.show()
    fig.savefig('./result/result_total/CFG_GMM_samples_3.png', dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("ok")




#######################################################################
# Train MLP-based Conditional Vector Field to target Gaussian mixture #
#######################################################################

def train_cfg_gmm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize GMM
    angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    means = 2 * torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
    covs = torch.tensor([0.2, 0.2, 0.2])
    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    gmm = GMM(means, covs, weights).to(device)

    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data=gmm,
        p_simple_shape=[2],
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)
    vector_field = MLPConditionalVectorField(
        dim=2,
        hidden_dim=256,
        class_dim=2,
        num_classes=3
    ).to(device)

    # Train vector field
    trainer = CFGTrainer(
        path=path,
        eta=0.25,
        null_label=3,
    )
    losses, steps = trainer.train(model=vector_field, num_steps=3000, lr=1e-3, batch_size=250)
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    # plt.show()
    plt.savefig('./result/result_total/CFG_GMM_loss_3.png', dpi=120, bbox_inches="tight")
    plt.close()
    print("ok")

    # Visualize Results
    visualize_cfg_gmm(gmm, vector_field, path, device)

def main():
    train_cfg_gmm()

if __name__ == "__main__":
    main()