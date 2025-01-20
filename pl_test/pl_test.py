import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MyModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # 从批次数据中分离输入和目标
        y_hat = self(x)  # 前向传播，获取预测输出
        loss = nn.functional.mse_loss(y_hat, y)  # 计算损失
        self.log('train_loss', loss)  # 记录损失
        return loss  # 返回损失，以供优化器使用

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# 准备数据
x_train = torch.randn(100, 10)  # 100个样本，10个特征
y_train = torch.randn(100, 1)    # 100个目标输出
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32)

# 训练模型
model = MyModel(input_dim=10, output_dim=1)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader)
