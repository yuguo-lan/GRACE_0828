import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os
import json
from typing import Optional, List

# 尝试导入 matplotlib，若未安装则跳过绘图
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed, skipping heatmap generation.")


class DataManager:
    """
    数据管理器：加载数据集，进行IID或Non-IID划分，为每个客户端分配数据。
    支持从预先生成的分区文件加载划分（若 partition_dir 指定且存在），
    否则动态生成并保存。
    """
    def __init__(
        self,
        dataset_name: str,
        num_clients: int,
        non_iid_alpha: Optional[float] = None,
        seed: int = 42,
        data_dir: str = "./data",
        partition_dir: Optional[str] = None,  # 新增：指定分区文件目录
        force_generate: bool = False,
    ):
        self.dataset_name = dataset_name.lower()
        self.num_clients = num_clients
        self.non_iid_alpha = non_iid_alpha
        self.seed = seed
        self.data_dir = data_dir
        self.transform_train, self.transform_test = self._get_transforms()
        self.train_dataset, self.test_dataset = self._load_datasets()
        self.force_generate = force_generate

        # 确定分区目录
        if partition_dir is None:
            alpha_str = f"alpha{self.non_iid_alpha}" if self.non_iid_alpha is not None else "IID"
            self.partition_dir = os.path.join(
                data_dir, "partitions", self.dataset_name,
                f"{self.num_clients}_{alpha_str}_seed{self.seed}"
            )
        else:
            self.partition_dir = partition_dir

        self.client_datasets = []      # 每个客户端的训练集（Subset）
        self.client_data_sizes = []    # 每个客户端的数据量
        self.client_label_dist = []    # 每个客户端的标签分布（可选）

        np.random.seed(seed)
        self._split_data()

    def _get_transforms(self):
        if self.dataset_name == "mnist":
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.dataset_name == "fashionmnist":
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        elif self.dataset_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif self.dataset_name == "cifar100":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return transform_train, transform_test

    def _load_datasets(self):
        if self.dataset_name == "mnist":
            train = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform_train)
            test = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transform_test)
        elif self.dataset_name == "fashionmnist":
            train = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=self.transform_train)
            test = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=self.transform_test)
        elif self.dataset_name == "cifar10":
            train = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=self.transform_train)
            test = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=self.transform_test)
        elif self.dataset_name == "cifar100":
            train = datasets.CIFAR100(self.data_dir, train=True, download=True, transform=self.transform_train)
            test = datasets.CIFAR100(self.data_dir, train=False, download=True, transform=self.transform_test)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return train, test

    def _split_data(self):
        
        if not self.force_generate and self._load_partition():
            return

        # 否则动态生成划分
        targets = np.array(self.train_dataset.targets)
        num_classes = len(self.train_dataset.classes)
        num_samples = len(targets)

        if self.non_iid_alpha is None:
            # IID: 按类别平均分配
            client_indices = [[] for _ in range(self.num_clients)]

            for k in range(num_classes):
                class_indices = np.where(targets == k)[0]
                np.random.shuffle(class_indices)

                # 每一类尽量平均分到每个客户端
                splits = np.array_split(class_indices, self.num_clients)
                for i in range(self.num_clients):
                    client_indices[i].extend(splits[i].tolist())

            # 打乱每个客户端内部顺序
            for i in range(self.num_clients):
                np.random.shuffle(client_indices[i])
        else:
            # Non-IID: Dirichlet分布
            min_samples_per_client = 1
            client_indices = [[] for _ in range(self.num_clients)]
            for k in range(num_classes):
                class_indices = np.where(targets == k)[0]
                np.random.shuffle(class_indices)
                proportions = np.random.dirichlet(np.repeat(self.non_iid_alpha, self.num_clients))
                proportions = proportions / proportions.sum()

                # Convert fractional Dirichlet allocations to integer counts
                # while preserving the exact class sample count. Using repeated
                # round() can over-allocate before the final client when N is
                # large or alpha is small.
                raw_counts = proportions * len(class_indices)
                counts = np.floor(raw_counts).astype(int)
                remainder = int(len(class_indices) - counts.sum())
                if remainder > 0:
                    frac_order = np.argsort(-(raw_counts - counts), kind="stable")
                    counts[frac_order[:remainder]] += 1

                assigned = 0
                for i, num in enumerate(counts.tolist()):
                    client_indices[i].extend(class_indices[assigned:assigned + num].tolist())
                    assigned += num
                if assigned != len(class_indices):
                    raise RuntimeError("Dirichlet partition did not preserve class sample count")

            # 确保每个客户端至少有一个样本
            for i in range(self.num_clients):
                if len(client_indices[i]) == 0:
                    for j in range(self.num_clients):
                        if len(client_indices[j]) > min_samples_per_client:
                            client_indices[i].append(client_indices[j].pop())
                            break
                    else:
                        raise RuntimeError("Unable to assign at least one sample to each client.")

        # 构建客户端数据集
        self._build_client_datasets(client_indices, targets, num_classes)

        # 保存划分到文件
        self._save_partition(client_indices, targets, num_classes)

        # 输出统计信息并绘制热力图（可选）
        self._print_stats(client_indices, targets, num_classes)

    def _load_partition(self) -> bool:
        """尝试从 self.partition_dir 加载预先生成的分区文件，返回是否成功"""
        config_path = os.path.join(self.partition_dir, "config.json")
        if not os.path.exists(config_path):
            return False

        with open(config_path, 'r') as f:
            config = json.load(f)

        # 比较整数/字符串参数
        if (config.get('dataset') != self.dataset_name or
            config.get('num_clients') != self.num_clients or
            config.get('seed') != self.seed):
            print(f"Warning: Partition config mismatch. Regenerating...")
            return False

        # 浮点数 alpha 近似比较
        saved_alpha = config.get('non_iid_alpha')
        current_alpha = self.non_iid_alpha
        if saved_alpha is None and current_alpha is not None:
            print(f"Warning: Partition config mismatch (alpha: None vs {current_alpha}). Regenerating...")
            return False
        if saved_alpha is not None and current_alpha is None:
            print(f"Warning: Partition config mismatch (alpha: {saved_alpha} vs None). Regenerating...")
            return False
        if saved_alpha is not None and current_alpha is not None:
            # 容差 1e-6
            if abs(saved_alpha - current_alpha) > 1e-6:
                print(f"Warning: Partition config mismatch (alpha: {saved_alpha} vs {current_alpha}). Regenerating...")
                return False

        # 加载每个客户端的索引
        client_indices = []
        targets = np.array(self.train_dataset.targets)
        num_classes = len(self.train_dataset.classes)

        for i in range(self.num_clients):
            npz_path = os.path.join(self.partition_dir, f"client_{i}.npz")
            if not os.path.exists(npz_path):
                return False
            data = np.load(npz_path)
            indices = data['indices']
            client_indices.append(indices.tolist())

        self._build_client_datasets(client_indices, targets, num_classes)
        print(f"Loaded partition from {self.partition_dir}")
        return True

    def _save_partition(self, client_indices, targets, num_classes):
        """将划分保存到 self.partition_dir"""
        os.makedirs(self.partition_dir, exist_ok=True)

        # 保存每个客户端的索引
        for i, indices in enumerate(client_indices):
            np.savez_compressed(os.path.join(self.partition_dir, f"client_{i}.npz"), indices=np.array(indices))

        # 保存配置
        config = {
            'dataset': self.dataset_name,
            'num_clients': self.num_clients,
            'non_iid_alpha': self.non_iid_alpha,
            'seed': self.seed,
            'num_classes': num_classes,
        }
        with open(os.path.join(self.partition_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Partition saved to {self.partition_dir}")

    def _build_client_datasets(self, client_indices, targets, num_classes):
        """根据索引列表构建 client_datasets, client_data_sizes, client_label_dist"""
        for indices in client_indices:
            subset = Subset(self.train_dataset, indices)
            self.client_datasets.append(subset)
            self.client_data_sizes.append(len(indices))
            if len(indices) > 0:
                labels = targets[indices]
                dist = np.bincount(labels, minlength=num_classes) / len(indices)
            else:
                dist = np.zeros(num_classes)
            self.client_label_dist.append(dist)

    def _print_stats(self, client_indices, targets, num_classes):
        """打印每个客户端的类别分布，并绘制热力图（如果 matplotlib 可用）"""
        # 构建计数矩阵
        counts = np.zeros((self.num_clients, num_classes), dtype=int)
        for client_id, indices in enumerate(client_indices):
            labels = targets[indices]
            for c in range(num_classes):
                counts[client_id, c] = np.sum(labels == c)

        # 打印到控制台
        print("\n" + "="*60)
        print("Client Data Distribution (per client, per class):")
        print(f"Dataset: {self.dataset_name}, Clients: {self.num_clients}, Alpha: {self.non_iid_alpha}")
        for client_id in range(self.num_clients):
            non_zero = [(c, counts[client_id, c]) for c in range(num_classes) if counts[client_id, c] > 0]
            print(f"Client {client_id:3d}: {non_zero}")
        print("="*60 + "\n")

        # 绘制热力图
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.figure(figsize=(12, 8))
                im = plt.imshow(counts, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                plt.colorbar(im, label='Number of samples')
                plt.xlabel('Class Label')
                plt.ylabel('Client ID')
                plt.title(f'Client Data Distribution\nDataset: {self.dataset_name.upper()}, Alpha: {self.non_iid_alpha}')
                plt.tight_layout()
                # 保存图片
                os.makedirs('data_split_results', exist_ok=True)
                alpha_str = f"alpha{self.non_iid_alpha}" if self.non_iid_alpha is not None else "IID"
                filename = f"data_split_results/data_distribution_{self.dataset_name}_{self.num_clients}clients_{alpha_str}.png"
                plt.savefig(filename, dpi=150)
                print(f"Heatmap saved to {filename}")
                plt.close()
            except Exception as e:
                print(f"Failed to generate heatmap: {e}")
        else:
            print("Skipping heatmap (matplotlib not installed).")

    # 其他公共方法保持不变
    def get_client_dataloader(self, client_id: int, batch_size: int, shuffle=True):
        return DataLoader(self.client_datasets[client_id], batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self, batch_size: int):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_client_info(self, client_id: int):
        return {
            'data_size': self.client_data_sizes[client_id],
            'label_dist': self.client_label_dist[client_id] if hasattr(self, 'client_label_dist') else None
        }