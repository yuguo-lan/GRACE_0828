import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """多层感知机，输入维度根据数据集自动计算"""
    def __init__(self, input_dim, hidden_dim=200, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """简单 CNN，使用自适应池化以支持不同输入尺寸"""
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_model_fn(model_name: str, dataset_name: str = 'MNIST'):
    """
    返回一个无参数的模型工厂函数，根据数据集自动配置模型细节。
    """
    dataset_upper = dataset_name.upper()
    # 根据数据集获取类别数、输入通道数、输入尺寸
    if dataset_upper in ['MNIST', 'FASHIONMNIST']:
        num_classes = 10
        input_channels = 1
        input_size = 28
    elif dataset_upper == 'CIFAR10':
        num_classes = 10
        input_channels = 3
        input_size = 32
    elif dataset_upper == 'CIFAR100':
        num_classes = 100
        input_channels = 3
        input_size = 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model_name = model_name.lower()
    if model_name == 'mlp':
        input_dim = input_channels * input_size * input_size
        return lambda: MLP(input_dim=input_dim, num_classes=num_classes)
    elif model_name == 'simplecnn':
        return lambda: SimpleCNN(input_channels=input_channels, num_classes=num_classes)
    elif model_name == 'resnet18':
        from torchvision.models import resnet18

        def create_resnet18():
            model = resnet18(weights=None, num_classes=num_classes)
            # CIFAR-style stem for 28x28/32x32 inputs. This is the standard
            # small-image adaptation and avoids the aggressive ImageNet stem.
            if input_size <= 32:
                model.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                model.maxpool = nn.Identity()
            elif input_channels != 3:
                model.conv1 = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            return model

        return create_resnet18
    else:
        raise ValueError(f"Unknown model: {model_name}")