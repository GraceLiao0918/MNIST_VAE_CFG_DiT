
from script_utils import *

from MNIST_data import MNISTSampler

from VAE import VAE

from DiT import DiffusionTransformerFlowModel

from CFG_GMM import CFGVectorFieldODE

from Adapter import attach_adapters

from digit_classifier import train_classifier,to_pixels

import os
import torch
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




@torch.no_grad()
def evaluate_models(models, vae, classifier,guidance_scale,samples_per_class,num_timesteps):

    device = next(vae.parameters()).device

    vae.eval()
    classifier.to(device)
    classifier.eval()

    for model in models.values():
        model.to(device)
        model.eval()

    # 每个数字生成samples_per_class张
    samples_per_class = samples_per_class
    num_timesteps = num_timesteps
    guidance_scale = guidance_scale

    output_dir = "./result/classifier_compare"
    os.makedirs(output_dir, exist_ok=True)

    # 记录每个模型生成正确的图片数量。
    correct_count = {"dit": 0, "adapter": 0}

    # 记录生成图片，最后拼图保存。
    generated_images = {"dit": [], "adapter": []}

    # 使用独立的随机数生成器，固定初始噪声。
    generator = torch.Generator().manual_seed(42)

    # 依次要求模型生成数字 0、1、2……9。
    for digit in range(10):

        # 例如 digit=5，这一批的目标标签就是 20 个 5。
        labels = torch.full(
            (samples_per_class,),
            digit,
            dtype=torch.long,
            device=device,
        )

        # 两个模型使用完全相同的初始噪声。
        z0 = torch.randn(
            samples_per_class, 128, 4, 4,
            generator=generator,
        ).to(device)

        # 创建 Euler 采样的时间点。
        ts = torch.linspace(0, 0.999, num_timesteps).to(device)
        ts = ts.view(1, num_timesteps, 1, 1, 1)
        ts = ts.expand(samples_per_class, -1, -1, -1, -1)

        for name, model in models.items():

            # Step 1: 从噪声生成潜变量。
            ode = CFGVectorFieldODE(
                model,
                guidance_scale=guidance_scale,
                null_label=10,
            )
            simulator = EulerSimulator(ode)

            z1 = simulator.simulate(
                z0.clone(),
                ts,
                y=labels,
                use_tqdm=False,
            )

            # Step 2: VAE 将潜变量解码为数字图片。
            images, _ = vae.decode(z1)

            # 与分类器训练时保持相同的预处理。
            images = to_pixels(images)

            # Step 3: 分类器预测每张图片是什么数字。
            scores = classifier(images)
            predictions = scores.argmax(dim=1)

            # Step 4: 比较预测数字和目标数字。
            correct = (predictions == labels).sum().item()
            correct_count[name] += correct

            generated_images[name].append(images.cpu())

            print(
                f"目标数字 {digit}，模型 {name}："
                f"{correct}/{samples_per_class} 张正确，"
                f"命中率 {correct / samples_per_class:.2%}"
            )

    # 统计全部数字的总体命中率。
    total = 10 * samples_per_class
    accuracy = {}

    for name in models:
        accuracy[name] = correct_count[name] / total

        # 拼接各批图片。每一行对应一个目标数字。
        images = torch.cat(generated_images[name], dim=0)
        save_image(
            images,
            os.path.join(output_dir, f"{name}_output.png"),
            nrow=samples_per_class,
        )

    print(f"\n原 DiT 命中率：{accuracy['dit']:.2%}")
    print(f"Adapter 命中率：{accuracy['adapter']:.2%}")

    difference = accuracy["adapter"] - accuracy["dit"]
    print(f"Adapter 提升：{difference * 100:+.2f} 个百分点")

    return accuracy


def compare_models():
    img_size = 4

    dit = DiffusionTransformerFlowModel(
        img_size=img_size,
        patch_size=1,
        num_layers=8,
        c=128,
        dim=256,
        heads=8,
        final_dim=10,
        n_classes=11,
    )
    dit_ckpt_path = "runs/white-jackfruit-8bda7561/step_  5000_model.pt"
    dit.load_state_dict(torch.load(dit_ckpt_path, map_location="cpu", weights_only=True))
    dit.to(device)
    dit.eval()

    adapter = DiffusionTransformerFlowModel(
        img_size=img_size,
        patch_size=1,
        num_layers=8,
        c=128,
        dim=256,
        heads=8,
        final_dim=10,
        n_classes=11,
    )
    adapter_ckpt_path = "runs/spring-tangerine-c7fd6ed8/step_  5000_adapter_model.pt"
    attach_adapters(adapter, width=16)
    adapter.load_state_dict(torch.load(adapter_ckpt_path, map_location="cpu", weights_only=True))
    adapter.to(device)
    adapter.eval()

    models = {"dit": dit, "adapter": adapter}

    vae = VAE(
        data_channels=1,
        hidden_channels=[16, 32, 64, 128],
        beta=10.0,
    )

    # 加载权重 训练1000步----Problem
    vae_ckpt_path = "runs/delicate-orange-f3910e8f/step_005000_model.pt"
    vae.load_state_dict(torch.load(vae_ckpt_path, map_location="cpu", weights_only=True))
    vae.to(device)
    vae.eval()

    epochs = 3  # 训练轮数

    test_set = datasets.MNIST(
        root='./data',
        train=False,  # 加载测试集
        download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1305,), (0.2891,)),
        ])
    )

    classifier, accuracy = train_classifier(
            test_set=test_set,
            data_root='./data',
            device=device,
            seed=42,
            epochs=epochs,
        )
    print(f"分类器在真实MNIST测试集上的准确率: {accuracy * 100:.2f}%")
    classifier.to(device).eval()

    evaluate_models(
        models=models,
        vae=vae,
        classifier=classifier,guidance_scale=5.0,samples_per_class=40,num_timesteps=100
    )


def main():
    compare_models()

if __name__ == "__main__":
    main()

