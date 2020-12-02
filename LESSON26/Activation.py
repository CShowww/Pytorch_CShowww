import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F

#sigmoid
def sigmoid(x):
    return (1/(1+np.exp(-x)))

x = np.linspace(-100,100,10)
y = [sigmoid(i) for i in x]
plt.plot(x,y,c='b',label = 'sigmoid')
plt.title('$Sigmoid$')
plt.legend()
plt.show()

a = torch.linspace(-100,100,10)
print(torch.sigmoid(a))

#Tanh
def tanh(x):
    return((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))

x = np.linspace(-3,3,100)
y = [tanh(i) for i in x]
plt.plot(x,y,c='b',label = 'Tanh')
plt.title('tanh')
plt.legend()
plt.show()

a = torch.linspace(-100,100,10)
print(torch.tanh(a))


#relu
def relu(x):
    return max(x,0)
x = np.linspace(-3,3,100)
y = [relu(i) for i in x]
plt.plot(x,y,c='b',label = 'relu')
plt.title('$relu$')
plt.legend()
plt.show()

'''
Mean Sqaured Error
Cross Entropy Loss
  binary
  multi-class
  softmax
Gradient API
  torch.autograd.grad(loss,[w])
  loss.backward()
'''

x = torch.ones(1)
w = torch.full([1],2)
w.requires_grad_()
mse = F.mse_loss(x*w,torch.ones(1))
print(torch.autograd.grad(mse,[w],retain_graph=True))

x = torch.ones(1)
w = torch.full([1],2)
w.requires_grad_()
mse = F.mse_loss(x*w,torch.ones(1))
mse.backward()
print(w.grad)

#softmax
a = torch.rand(3)
a.requires_grad_()
p = F.softmax(a,dim = 0)
print(torch.autograd.grad(p[1],[a],retain_graph=True))
print(torch.autograd.grad(p[2],[a],retain_graph=True))

