import torch
from torch_geometric.data import Data

# 节点特征，一个包含3个节点，每个节点1个特征的张量
x = torch.tensor([[1], [2], [3]], dtype=torch.float)

# 边索引，表示节点之间的连接
edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)  # 0->1, 1->0, 1->2

# 创建 Data 对象
data = Data(x=x, edge_index=edge_index)

# 输出相关信息
print(data)
print(data.x)          # 节点特征
print(data.edge_index) # 边索引
