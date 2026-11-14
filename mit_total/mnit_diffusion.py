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

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: b d
        """
        pass

class LabeledSampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: b d
            - labels: b
        """
        pass

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: b ...
            - t: b
        Returns:
            - drift_coefficient: b ...
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: b ...
            - t: b
        Returns:
            - drift_coefficient: b ...
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: b ...
            - t: b
        Returns:
            - diffusion_coefficient: b ...
        """
        pass

class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,)), torch.zeros(1,)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,)), torch.ones(1,)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: b
        Returns:
            - alpha_t: b
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: b
        Returns:
            - d/dt a_t: b
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)

class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1)), torch.ones(1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1)), torch.zeros(1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: b
        Returns:
            - beta_t: b
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: b
        Returns:
            - d/dt beta_t: b
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)

class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: b
        Returns:
            - alpha_t: b
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: b
        Returns:
            - d/dt alpha_t b
        """
        return torch.ones_like(t)

class LinearBeta(Beta):
    """
    Implements beta_t = 1-t
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: b
        Returns:
            - beta_t: b
        """
        return 1-t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: b
        Returns:
            - d/dt alpha_t: b
        """
        return - torch.ones_like(t)

class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)

class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: LabeledSampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: b
        Returns:
            - x: samples from p_t(x), b ... (i.e.,. `b d`, `b c h w`, etc.)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples) # (b ...)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (b ...)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: b ...
            - y: b
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable b ...
            - t: time b
        Returns:
            - x: samples from p_t(x|z), b ...
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: b ...
            - z: b ...
            - t: b
        Returns:
            - conditional_vector_field: conditional vector field b c h w
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: b ...
            - z: b ...
            - t: b
        Returns:
            - score: b ...
        """
        pass




class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
        self.rearrange_scalar = Rearrange(f'b -> b{" 1" * len(p_simple_shape)}')

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: b ...
            - y: b
        """
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: b ...
            - t: b
        Returns:
            - x: b ...
        """
        alpha_t = self.rearrange_scalar(self.alpha(t)) # (b 1 1 1)
        beta_t = self.rearrange_scalar(self.beta(t)) # (b 1 1 1)
        return alpha_t * z + beta_t * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: b c h w
            - z: b c h w
            - t: b
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        alpha_t = self.rearrange_scalar(self.alpha(t)) # b
        beta_t = self.rearrange_scalar(self.beta(t)) # b
        dt_alpha_t = self.rearrange_scalar(self.alpha.dt(t)) # b
        dt_beta_t = self.rearrange_scalar(self.beta.dt(t)) # b

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: b ...
            - z: b ...
            - t: b
        Returns:
            - conditional_score: b ...
        """
        alpha_t = self.rearrange_scalar(self.alpha(t))
        beta_t = self.rearrange_scalar(self.beta(t))
        return (z * alpha_t - x) / beta_t ** 2

class MNISTSampler(nn.Module, LabeledSampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self):
        super().__init__()
        # 下载数据集MNIST
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1305,), (0.2891,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        # 数据处理
        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels

# # Change these!
# num_rows = 3
# num_cols = 3
# num_timesteps = 5
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Initialize our sampler
# sampler = MNISTSampler().to(device)
#
# # Initialize probability path
# path = GaussianConditionalProbabilityPath(
#     p_data = MNISTSampler(),
#     p_simple_shape = [1, 32, 32],
#     alpha = LinearAlpha(),
#     beta = LinearBeta()
# ).to(device)
#
# # Sample
# num_samples = num_rows * num_cols
# z, _ = path.p_data.sample(num_samples)
# z = z.view(-1, 1, 32, 32)
#
# # Setup plot
# fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))
#
# # Sample from conditional probability paths and graph
# ts = torch.linspace(0, 1, num_timesteps).to(device)
# for tidx, t in enumerate(ts):
#     tt = t.expand(num_samples) # b
#     xt = path.sample_conditional_path(z, tt) # b 1 32 32
#     grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
#     axes[tidx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
#     axes[tidx].axis("off")
# plt.show()



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






# 高斯混合模型
class GMM(nn.Module, LabeledSampleable):
  def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor):
    super().__init__()
    self.means = nn.Buffer(means)
    self.covariances = nn.Buffer(covariances)
    self.weights = nn.Buffer(weights)

  def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      - num_samples: the desired number of samples
    Returns:
      - samples: b n
      - labels: b
    """
    # Choose the amount of each mode
    # Perform multinomial sampling on CPU to avoid device-side assert errors
    labels = torch.multinomial(self.weights.cpu(), num_samples=num_samples, replacement=True).to(self.means.device)

    # Sample from each mode
    samples = torch.zeros(num_samples, self.means.shape[1]).to(self.means.device)
    for idx in range(len(self.means)):
      samples[labels == idx] = torch.randn_like(samples[labels == idx]) * self.covariances[idx] + self.means[idx]

    return samples, labels

# 计算模型大小
MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(
        self,
        **kwargs
      ):
        super().__init__()
        self.model = None
        self.opt = None
        self.output_dir = None

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def checkpoint(self, step: int):
      pass

    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def random_name(self) -> str:
        adjectives = ["autumn", "hidden", "bitter", "misty", "silent", "empty", "dry", "dark", "summer", "icy", "delicate", "quiet", "white", "cool", "spring", "winter", "patient"]
        foods = ["apple", "banana", "pear", "plum", "orange", "persimmon", "tangerine", "durian", "jackfruit", "jicama", "cantaloupe", "watermelon", "peach"]
        return f"{random.choice(adjectives)}-{random.choice(foods)}-{str(uuid.uuid4())[:8]}"

    def train(
        self,
        model: nn.Module,
        num_steps: int,
        lr: float = 1e-3,
        warmup_steps: int = 500,
        ckpt_every: Optional[int] = 500,
        run_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[float], List[int]]:
        """
        Linear warmup from 0 -> lr over `warmup_steps`, then constant lr.
        """
        # Initialize run name and output directory
        run_name = run_name or self.random_name()
        self.output_dir = os.path.join("runs", run_name)
        os.makedirs(self.output_dir, exist_ok=False)
        print("Initialized output directory at: " + self.output_dir)

        # Grab size
        self.model = model
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        # Initialize optimizer and LR
        self.opt = self.get_optimizer(lr)
        self.model.train()

        for pg in self.opt.param_groups:
            pg["lr"] = 0.0

        # Main training loop
        losses: List[float] = []
        steps: List[int] = []

        pbar = tqdm(range(num_steps))
        for step in pbar:
            # Update LR
            if warmup_steps > 0 and step < warmup_steps:
                cur_lr = lr * float(step + 1) / float(warmup_steps)
            else:
                cur_lr = lr
            for pg in self.opt.param_groups:
                pg["lr"] = cur_lr

            # Forward + backward
            self.opt.zero_grad(set_to_none=True)
            loss = self.get_train_loss(**kwargs)
            loss.backward()

            # Take gradient step
            self.opt.step()

            losses.append(float(loss.detach().item()))
            steps.append(step)

            pbar.set_description(f"Step {step}, lr={cur_lr:.2e}, loss={loss.item():.4f}")

            # Callback if specified
            if ckpt_every is not None and step % ckpt_every == 0 and step > 0:
              self.model.eval()
              self.checkpoint(step)
              self.model.train()

        self.model.eval() # 评估模式
        return losses, list(range(num_steps))


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


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: b ...
            - t: b
            - dt: b
        Returns:
            - nxt: b ...
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: b ...
            - ts: b
        Returns:
            - x_final: b ...
        """
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: b ...
            - ts: b nt
        Returns:
            - x_traj: b nt ...
        """
        x_traj = [x.clone()]
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            x_traj.append(x.clone())
        return torch.stack(x_traj, dim=1)

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.sde.drift_coefficient(xt, t, **kwargs) * h + self.sde.diffusion_coefficient(xt, t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.sde.drift_coefficient(xt, t, **kwargs) * h + self.sde.diffusion_coefficient(xt, t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )


#######################################################################
# Train MLP-based Conditional Vector Field to target Gaussian mixture #
#######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize GMM
# angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
# means = 2 * torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
# covs = torch.tensor([0.2, 0.2, 0.2])
# weights = torch.tensor([1/3, 1/3, 1/3])
# gmm = GMM(means, covs, weights).to(device)
#
# # Initialize probability path
# path = GaussianConditionalProbabilityPath(
#     p_data = gmm,
#     p_simple_shape = [2],
#     alpha = LinearAlpha(),
#     beta = LinearBeta()
# ).to(device)
# vector_field = MLPConditionalVectorField(
#     dim = 2,
#     hidden_dim = 256,
#     class_dim = 2,
#     num_classes = 3
# ).to(device)
#
# # Train vector field
# trainer = CFGTrainer(
#     path=path,
#     eta=0.25,
#     null_label=3,
# )
# losses, steps = trainer.train(model=vector_field, num_steps=3000, lr=1e-3, batch_size=250)
# plt.plot(steps, losses)
# plt.xlabel("Step")
# plt.ylabel("Loss")
# plt.show()

#####################
# Visualize Results #
#####################

# # User knobs
# guidance_strength = 8.0 # try changing me!
#
# fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 6))
#
# # Panel 1: Target
# ax = axes[0]
# x_data, _ = gmm.sample(250)
# x_data = x_data.detach().cpu().numpy()
# ax.scatter(x_data[:, 0], x_data[:, 1], s=5, marker="*")
# ax.set_title("Target")
#
# # Panel 2: Conditioned on each mode
# ax = axes[1]
# cfg_vector_field = CFGVectorFieldODE(vector_field, guidance_scale=guidance_strength, null_label=3)
# simulator = EulerSimulator(cfg_vector_field)
#
# batch_size = 250
# labels = torch.arange(3).repeat_interleave(batch_size).to(device)
# x_init = path.p_simple.sample(3 * batch_size) # b 2
# ts = torch.linspace(0, 1, 100).expand(3 * batch_size, -1).to(device) # b nt
# xs = simulator.simulate(x_init, ts, y=labels) # b 2
# for idx in range(3):
#     xs_idx = xs[idx * batch_size: (idx + 1) * batch_size].detach().cpu().numpy()
#     ax.scatter(xs_idx[:, 0], xs_idx[:, 1], s=5, label=f"Mode {idx}", marker="*")
# ax.legend()
# ax.set_title(f"CFG w/ Guidance Strength {guidance_strength:.2f}")
#
# # Panel 3: Unconditioned
# ax = axes[2]
# batch_size = 750
# labels = torch.ones(batch_size).long().to(device) * 3
# x_init = path.p_simple.sample(batch_size) # b 2
# ts = torch.linspace(0, 1, 100).expand(batch_size, -1).to(device) # b nt
# xs = simulator.simulate(x_init, ts, y=labels).detach().cpu().numpy() # b 2
# ax.scatter(xs[:, 0], xs[:, 1], s=5, label=f"Mode {idx}", marker="*")
# ax.set_title(f"Unguided Samples")
# plt.show()


# 傅里叶时间编码器
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


# 分块器
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




# if __name__ == '__main__':


# #################
# # Training code #
# #################
#
# # Initialize our sampler
# sampler = MNISTSampler().to(device)
#
# # Initialize probability path
# path = GaussianConditionalProbabilityPath(
#     # p_data = MNISTSampler(),
#     p_data = sampler,
#     p_simple_shape = [1, 32, 32],
#     alpha = LinearAlpha(),
#     beta = LinearBeta()
# ).to(device)
#
# # Initialize model
# dit = DiffusionTransformerFlowModel(
#     img_size = 32,
#     patch_size = 4,
#     num_layers = 8,
#     c = 1,
#     dim = 256,
#     heads = 8,
#     final_dim = 10,
#     n_classes = 11,
# ).to(device)
#
# # Initialize trainer
# trainer = MNISTCFGTrainer(path = path, eta=0.35, null_label=10)
#
# # Train! You should have reasonable results in ~15 A100 minutes
# # losses, steps = trainer.train(model=dit, num_steps = 20000, lr=0.4e-3, batch_size=256, ckpt_every=1000)
# losses, steps = trainer.train(model=dit, num_steps = 1200, lr=0.2e-3, batch_size=64, ckpt_every=2000)
#
# plt.plot(steps, losses)
# plt.xlabel("Step")
# plt.ylabel("Loss")
# plt.title("Loss vs. Step")
# plt.show()
#
#
# # Play with these!
# samples_per_class = 10
# num_timesteps = 100
# guidance_scales = [1.0, 3.0, 5.0]
#
# visualize_output(
#     model=dit,
#     path=path,
#     samples_per_class=samples_per_class,
#     num_timesteps=num_timesteps,
#     guidance_scales=guidance_scales,
# )
# plt.show()
#
#



















