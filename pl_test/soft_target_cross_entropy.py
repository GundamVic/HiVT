import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 自定义损失类
class SoftTargetCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


# 简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 生成随机数据
def generate_random_data(num_samples, input_size, num_classes):
    X = torch.randn(num_samples, input_size)  # 特征数据
    Y = F.one_hot(torch.randint(0, num_classes, (num_samples,)), num_classes=num_classes).float()  # 软目标
    return X, Y


# 训练模型
def train_model(model, criterion, optimizer, data_loader, num_epochs):
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == "__main__":
    # 参数设置
    input_size = 20  # 输入特征的大小
    num_classes = 5  # 总类别数
    num_samples = 100  # 数据样本数
    num_epochs = 10  # 训练轮次
    batch_size = 10  # 批大小

    # 生成随机数据
    X, Y = generate_random_data(num_samples, input_size, num_classes)
    dataset = torch.utils.data.TensorDataset(X, Y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始模型、损失函数和优化器
    model = SimpleNN(input_size, num_classes)
    criterion = SoftTargetCrossEntropyLoss(reduction='mean')  # 使用自定义损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, criterion, optimizer, data_loader, num_epochs)
