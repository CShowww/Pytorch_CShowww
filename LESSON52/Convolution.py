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

#torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
layer = nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)
x = torch.rand(1,1,28,28)
out = layer.forward(x)
print(out.shape)

layer = nn.Conv2d(1,3,kernel_size=3,stride=2,padding=1)
out = layer.forward(x)
print(out.size())

print(layer.weight)
print(layer.weight.size())
print(layer.bias.size())


#F.conv2d
w = torch.rand(16,3,5,5)
b = torch.rand(16)
x = torch.rand(1,3,28,28)
out = F.conv2d(x,w,b,stride=1,padding=1)
print(out.size())
