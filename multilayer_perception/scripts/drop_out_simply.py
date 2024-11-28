import torch 
from torch import nn
from d2l import torch as d2l

'''
dropout为传入的概率
'''
def dropout_layer(X, dropout):
    assert 0 <= dropout <=1
    if dropout == 1:
        # 全部丢弃
        return torch.zeros_like(X)
    if dropout == 0:
        return  X
    mask = (torch.rand(X.shape) > dropout).float()
    # print(f"mask:{mask}")
    return mask * X / (1.0 - dropout)

dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
# 训练次数，学习率和样本数量
num_epochs, lr, batch_size = 10, 0.5, 256
# s损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 训练数据和测试数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=lr)
# 训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
