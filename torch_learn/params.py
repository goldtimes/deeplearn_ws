'''
参数管理，深度学习过程需要保存训练的参数，复用参数以及检查参数
我们学习：
    1. 如何访问参数，用于调试，诊断和可视化
    2. 参数初始化
    3. 不同模型组件间共享参数
'''

import torch 
from torch import nn

'''---------------访问参数----------------'''
# 定义模型
net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,1))
# 初始化2，4的张量
X = torch.rand(size=(2,4))
print(net(X))
# 因为使用sequentail定义的模型，于是我们可以通过下标访问任意的层
# 打印全连接层的权重和bias
print(net[2].state_dict())
# 参数并不是简单的数值，而是通过torch.nn.parameter.Paramter类来管理它，通过data访问具体的参数内容，还能访问它的梯度
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
# 一次性访问所以层的参数
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 嵌套块的参数访问
def block1():
    return nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range (4):
        net.add_module(f'block{i}',block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4,1))
print(rgnet(X))
print(rgnet)
'''---------------访问参数----------------'''
'''---------------参数初始化----------------'''
def init_normal(m):
    if type(m) == nn.Linear:
        # 权重正态分布
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # bias为0
        nn.init.zeros_(m.bias)
# 参数初始化
net.apply(init_normal)
# 访问参数
print(net[0].weight.data) 
print(net[0].bias.data)

# 初始化为常数的权重
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)

# 对不同层的参数初始化
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)

'''---------------参数初始化----------------'''
'''---------------参数绑定----------------'''
# 我们需要给共享层一个名称，以便可以引用它的参数
# 其实就是定义一个线性层，拿到它的返回值
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
# 修改参数后，两个层的参数都修改了
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
'''---------------参数绑定----------------'''