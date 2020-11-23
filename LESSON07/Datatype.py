import torch
import numpy as np

'''Typecheck'''
a = torch.randn(3,3)
print(a.type())
print(isinstance(a,torch.FloatTensor))




'''Dimension0'''
print(torch.tensor(1.))
print(torch.tensor(1.3))

a = torch.tensor(2.2)
print(a.shape)
print(len(a.shape))
print(a.size())

'''Dimension1'''
print(torch.tensor([1.1]))
print(torch.tensor([1.1,1.2]))
print(torch.FloatTensor(1))
print(torch.FloatTensor(2))

data = np.ones(2)
print(torch.from_numpy(data))
b = torch.ones(2)
print(b.shape)



'''Dimension2'''

a = torch.randn(2,3)
print(a)
print(torch.FloatTensor([2,3]))
print(a.shape)
print(a.size())
print(a.size(0))
print(a.size(1))


'''Dimension3'''
a = torch.rand(1,2,3)
print(a)
print(a[0])
print(list(a.shape))
print(a.dim())
print(a.numel())
