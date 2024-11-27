'''
对上一节的内容用torch来实现
torch封装了优化器,损失函数和迭代器等等
'''

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))
# 输入为2，输出为1
net = nn.Sequential(nn.Linear(2,1))
# 模型参数的初始化
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 损失函数
loss = nn.MSELoss()
# HuberLoss效果差很多
# loss = nn.HuberLoss()
# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # net(X) 前向传播，计算了预测的值
        # 计算损失函数
        l = loss(net(X) ,y)
        trainer.zero_grad()
        # 反向传播计算梯度
        l.backward()
        # 如何访问梯度
        # print(net[0].weight.grad)
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)