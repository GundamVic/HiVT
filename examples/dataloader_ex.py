import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 3)  # 100个样本
        self.labels = torch.randint(0, 2, (100,))  # 100个标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集实例
dataset = MyDataset()

# 创建 DataLoader
batch_size = 10
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 遍历 DataLoader
for batch_data, batch_labels in data_loader:
    print("Batch data:", batch_data)
    print("Batch labels:", batch_labels)
