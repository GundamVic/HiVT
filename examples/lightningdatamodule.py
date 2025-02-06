import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        super(MyDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # 下载数据集（只需运行一次）
        datasets.MNIST(root='.', train=True, download=True)
        datasets.MNIST(root='.', train=False, download=True)

    def setup(self, stage=None):
        # 根据阶段设置数据集
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST(root='.', train=True, transform=transforms.ToTensor())
            self.val_dataset = datasets.MNIST(root='.', train=False, transform=transforms.ToTensor())
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(root='.', train=False, transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# 使用自定义数据模块
data_module = MyDataModule(batch_size=64)
data_module.prepare_data()
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# 你可以用 train_loader 和 val_loader 去训练你的模型
