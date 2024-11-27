import torch

# 创建一个行向量
vec =  torch.arange(12)
print(vec)
# shape
print(vec.shape)
# reshape 得到一样的操作
print(vec.reshape(3,4))
print(vec.reshape(-1,4))
print(vec.reshape(3,-1))
# zero
zeros = torch.zeros((2,3,4))
# ones
ones = torch.ones((2,3,4))

print(zeros)
print(ones)