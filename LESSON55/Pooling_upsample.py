import torch
import torch.nn as nn
from torch.nn import functional as F


x = torch.rand(1,16,14,14)

#第一个参数是kernel大小
layer = nn.MaxPool2d(2,stride=2)
out = layer(x)
print(out.size())

out = F.avg_pool2d(x,2,stride = 2)
print(out.size())


#upsample

x = out
out =  F.interpolate(x,scale_factor = 2,mode = 'nearest')
print(out.shape)

layer = nn.ReLU(inplace = True)
out = layer(x)
print(out.shape)
