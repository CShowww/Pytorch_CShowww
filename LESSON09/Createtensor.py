import torch
import numpy as np

'''Import from numpy'''
a = np.array([2,3.3])
print(torch.from_numpy(a))

a = np.ones([2,3])
print(torch.from_numpy(a))


'''import from list'''
print(torch.tensor([2.,3.2]))
print(torch.FloatTensor([2.,3.2]))
print(torch.tensor([[1.,2.2],[2.,2.2]]))

'''uninitialized'''
print(torch.empty(1))
print(torch.Tensor(2,3))
print(torch.IntTensor(2,3))


'''setdefaultvalue'''
print(torch.tensor([1.2,3]).type())
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2,3]).type())

'''rand/rand_like/randint'''
'''rand[0,1],rande_like,randint[min,max)'''

print(torch.rand(3,3))

a = torch.rand(3,3)
print(torch.rand_like(a))

'''min,max,shape'''
print(torch.randint(1,10,(3,3)))

print(torch.randn(3,3))
print(torch.normal(mean = torch.full([10],0),std = torch.arange(1,0,-0.1)))


'''full'''
print(torch.full([2,3],7))
'''scalar'''
print(torch.full([],7))
'''vector'''
print(torch.full([1],7))


'''arange'''
'''[min,max)'''
print(torch.arange(0,10))
print(torch.arange(0,10,2))


'''linspace/logspace'''
'''[min,max]'''
print(torch.linspace(0,10,steps=2))
print(torch.linspace(0,10,steps=11))

print(torch.logspace(0,-1,steps=10))


'''ones,zeros,eye'''
print(torch.ones(3,3))
print(torch.zeros(3,3))
print(torch.eye(3,3))

a = torch.zeros(3,3)
print(torch.ones_like(a))


