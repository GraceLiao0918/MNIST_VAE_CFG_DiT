
# 分类器

# 仅用真实 MNIST 训练，用来判断生成图像的数字类别

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
from torchvision import datasets, transforms

from MNIST_data import MNISTSampler

def to_pixels(images):
    # 反归一化到[0, 1]灰度范围
    return (images * 0.2891 + 0.1305).clamp(0, 1)

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   #32*32-->16*16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  #16*16-->8*8
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(), #2048---->64维
        )
        self.head = nn.Linear(64, 10) #64维---->10 labels

    def forward(self, images):
        return self.head(self.encoder(images))

def train_classifier(test_set, *, data_root, device, seed, epochs):
    torch.manual_seed(seed)

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1305,), (0.2891,)),
    ])

    train_set = datasets.MNIST(
        root=str(data_root),
        train=True,  # 加载训练集
        download=True,  # 如果本地没有会自动下载
        transform=train_transform
    )

    loader = DataLoader(train_set, batch_size=256, shuffle=True)
    classifier = DigitClassifier().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3) #AdamW优化器
    criterion = nn.CrossEntropyLoss()  #交叉熵损失函数CrossEntropyLoss()
    for _ in trange(epochs, desc="MNIST classifier"):
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(classifier(to_pixels(images.to(device))), labels.to(device))
            loss.backward()
            optimizer.step()

    classifier.eval() #评估模式

    correct = 0
    with torch.inference_mode():
        for images, labels in DataLoader(test_set, batch_size=256):
            predictions = classifier(to_pixels(images.to(device))).argmax(dim=1) #预测概率最大的数字
            correct += predictions.cpu().eq(labels).sum().item()
    return classifier, correct / len(test_set)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = "./data"
    epochs = 3  #训练轮数

    # 测试集 (test_set)
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1305,), (0.2891,)),
    ])

    test_set = datasets.MNIST(
        root='./data',
        train=False,  # 加载测试集
        download=True,
        transform=test_transform
    )

    # 返回分类器和准确率
    classifier, accuracy = train_classifier(
        test_set=test_set,
        data_root=data_root,
        device=device,
        seed=42,
        epochs=epochs
    )

    print(f"\n" + "=" * 40)
    print(f"分类器在真实MNIST测试集上的准确率: {accuracy * 100:.2f}%")
    print(f"=" * 40)


if __name__ == "__main__":
    main()