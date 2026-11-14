# MNIST图像条件生成

> 课程：[Flow Matching and Diffusion Models — 2026 Version](https://diffusion.csail.mit.edu/2026/index.html)
> 
> 主题：MNIST 条件生成、Classifier-Free Guidance（CFG）、Diffusion Transformer（DiT）、VAE 与潜空间生成、Adapter。


## 1.  概述

将二维分布生成扩展到 MNIST 图像。图像是高维张量，实验以 `32 × 32` 灰度图像为例，并逐步构建三种生成方案：

1. **像素空间生成**：直接在 `1 × 32 × 32` 图像空间训练带类别条件的向量场。
2. **潜空间生成**：先训练 VAE，将图像压缩到小尺寸潜变量，再在潜空间训练条件生成模型，最后由 VAE 解码回图像。
3. Adapter微调：在训练好的潜空间生成模型上进行Adapter微调

方案都使用条件 Flow Matching，并用 CFG 在采样时增强类别条件。

```text
MNIST 图像 x、类别 y
        │
        ├── 像素空间：Gaussian probability path → DiT → 生成图像
        │
        └── VAE 编码器 → 潜变量 z → latent DiT → VAE 解码器 → 生成图像
```

详细介绍见博客：https://graceliao0918.github.io/Projects/MNIST_Adapter_CFG_VAE_DiT

## 2.  实验

- 运行.py文件即可
- 模型需要自己训练，文件太大，无法上传
