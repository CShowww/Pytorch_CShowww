'''
Convolution
Input_channels
kernel_channels
kernel_size
stride
padding
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

#torch.nn.Conv1d(input_channels, number of kernel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
layer = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)
x = torch.rand(1,1,28,28)
#一次卷积运算
out = layer.forward(x)#一般不这么用，一般用out = layer(x)
print(out.shape)

layer = nn.Conv2d(1,3,kernel_size=3,stride=2,padding=1)
out = layer.forward(x)
print(out.size())

#weight第一个维度是weight的数量（kernel数量），因为是三个channel，所有就是三，第二个维度是input的channel数量
#后面的两个维度是kernel的size
#bias取决于kernel的数量
print(layer.weight)
print(layer.weight.size())
print(layer.bias.size())


#F.conv2d
w = torch.rand(16,3,5,5)
b = torch.rand(16)
x = torch.rand(1,3,28,28)
out = F.conv2d(x,w,b,stride=1,padding=1)
print(out.size())
