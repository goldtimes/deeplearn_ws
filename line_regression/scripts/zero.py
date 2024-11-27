'''
线性回归的从零开始实现
我们提供一个真实值的线性模型w,b，对数据+噪声
然后用深度学习的模型拟合出w,b
'''
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_example):
    # 生成正态分布的数据
    X = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(X, w) + b
    # 产生相同的维度的噪声
    y += torch.normal(0,0.01, y.shape)
    return X, y.reshape(-1, 1)

def vis(features, labels):
    d2l.set_figsize()
    # print(features[:, (1)].detach().numpy())
    # print(labels.detach().numpy())
    d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序,打乱了indices中的排序
    random.shuffle(indices)
    # 遍历所有样本，步长为batch_size
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[ i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def test_data_iter(features, labels):
    batch_size = 10

    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

# 定义模型
def linreg(X, w, b):
    # 线性回归模型
    return torch.matmul(X,w) + b

# 定义损失函数
'''
y_hat 预测值
y 真实值
'''
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 定义优化算法 梯度下降算法
'''
params 模型参数
lr 学习率 每 一步更新的大小由学习速率lr决定
batch_size 所以我们用批量大小（batch_size） 来规范化步长
'''
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 初始化参数
#   重复训练
        # 计算梯度  损失函数对模型w,b的偏导数
        # 更新参数

if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # print(features)
    # vis(features, labels)
    # test_data_iter(features, labels)
    # 初始化w,b
    w = torch.normal(0,0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.03
    num_epoch3 = 3
    net = linreg
    loss = squared_loss
    batch_size = 10
    for epoch in range(num_epoch3):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            # 打印每一次的loss
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')   
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')