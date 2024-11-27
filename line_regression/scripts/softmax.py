import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 初始化模型参数
# 因为数据集是28x28的图像，输出是10类所以输入是 28*28 
num_input = 784
num_output = 10
W = torch.normal(0,0.01, size=(num_input, num_output), requires_grad=True)
b = torch.zeros(num_output, requires_grad=True)
lr = 0.1
num_epochs = 10

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        # print(args)
        # zip将size相同的可迭代对象拿出来，然后遍历list中的每个元素 
        # data[0,0] args[1.0,2.0]/args[2.0,2.0]/args[0.0,2.0]
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        # data = [1.0, 2.0]/[2.0,2.0] data[0] 就是正确的个数，2.0为样本的个数
        # print(self.data)

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def sum_test():
    # 2x3的矩阵
    X = torch.tensor([[1.0, 2.0, 3.0],[4.0,5.0,6.0]])
    X_sum0 =  X.sum(0, keepdim=True) # 沿0维就会把0维消掉
    X_sum1 = X.sum(1, keepdim=True)
    print(X_sum0.shape)
    print(X_sum1.shape)
    print(softmax(X))


#  矩阵中的非常大或非常小的元素可能造成数值上溢或下溢，
def softmax(X):
    # 对每个项求了幂
    X_exp = torch.exp(X)
    # 789*10 的矩阵，对第1维求和
    parition = X_exp.sum(1, keepdim=True)
    # print(parition.shape)
    return X_exp / parition


# 定义模型
def net(X):
    # return softmax(torch.matmul(X.))
    # X(10,789) W(789, 10)
    linge = torch.matmul(X.reshape((-1, W.shape[0])), W) + b
    return softmax(linge)

# 定义交叉商的损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            # y.numel() 元素的总数
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 取出预测中的概率最大值的下标 这里拿到2,2
        y_hat = y_hat.argmax(axis=1)
        # print(f"y_hat arg max:{y_hat}")
    # 但是y[0,2]为真实值
    cmp = y_hat.type(y.dtype) == y
    # print(cmp) # tensor([False,  True])
    result = float(cmp.type(y.dtype).sum()) # 1.0

    metric = Accumulator(2)  # 正确预测数、预测总数
    metric.add(result, y.numel())
    # print(metric[0] / metric[1])
    return result

def cross_entropy_test():
    # 真实的标签
    y = torch.tensor([0, 2])
    # 预测样本得到的概率值
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # 0行的第0个元素0.1
    # 1行的第2个元素0.5
    # print(y_hat[[0, 1], y])
    # print(cross_entropy(y_hat, y))
    # 计算正确率
    accuracy(y_hat, y) / len(y)


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): 
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    # 训练
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    print(train_loss)
    print(train_acc)
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def main():
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()




if __name__ == "__main__":
    # sum_test()
    # cross_entropy_test()
    main()