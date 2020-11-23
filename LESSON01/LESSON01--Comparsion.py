import torch
from torch import autograd
import time

x = torch.tensor(2.)
a = torch.tensor(1.,requires_grad= True)
b = torch.tensor(2.,requires_grad= True)
c = torch.tensor(3.,requires_grad= True)

y = a**2*x + b*x +c
print('before:',a.grad,b.grad,c.grad)
grads = autograd.grad(y,[a,b,c])
print('after:',grads[0],grads[1],grads[2])


print(torch.__version__)
print(torch.cuda.is_available())

a = torch.randn(10000,1000)
b = torch.randn(1000,2000)

t0 = time.time()
c = torch.matmul(a,b)
t1 = time.time()
print (a.device,t1-t0,c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)


t0 = time.time()
c = torch.matmul(a,b)
t1 = time.time()
print('Initialization:',a.device,t1-t0,c.norm(2))

t0 = time.time()
c = torch.matmul(a,b)
t1 = time.time()
print('Normal:',a.device,t1-t0,c.norm(2))