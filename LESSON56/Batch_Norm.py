'''
BatchNorm
LayerNorm
InstanceNorm
GroupNorm
'''

import torch
import torch.nn as nn
x = torch.rand(100,16,784)
layer = nn.BatchNorm1d(16)
out = layer(x)
print(layer.running_mean)
print(layer.running_var)


x = torch.rand(1,16,7,7)
layer = nn.BatchNorm2d(16)
out = layer(x)
print(layer.weight)
print(layer.bias)
print(vars(layer))