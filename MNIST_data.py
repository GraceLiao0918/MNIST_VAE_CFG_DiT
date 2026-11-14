# 数据处理


from script_utils import *


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


# Visualize
def visualize_mnist_conditionalgaussianprobabilitypath():
    num_rows = 3
    num_cols = 3
    num_timesteps = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize our sampler
    sampler = MNISTSampler().to(device)

    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data = MNISTSampler(),
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Sample
    num_samples = num_rows * num_cols
    z, _ = path.p_data.sample(num_samples)
    z = z.view(-1, 1, 32, 32)

    # Setup plot
    fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))

    # Sample from conditional probability paths and graph
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for tidx, t in enumerate(ts):
        tt = t.expand(num_samples) # b
        xt = path.sample_conditional_path(z, tt) # b 1 32 32
        grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
        axes[tidx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[tidx].axis("off")
    # plt.show()
    # plt.show = save_preview

    fig.savefig('./result/result_total/mnist_data_sample.png', dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("ok")

def main():
    visualize_mnist_conditionalgaussianprobabilitypath()

if __name__ == "__main__":
    main()