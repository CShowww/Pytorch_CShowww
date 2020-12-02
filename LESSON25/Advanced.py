import torch

'''where
gather
'''

#where
cond = torch.rand(2,2)
print(cond)
a = torch.zeros(2,2)
b = torch.ones(2,2)

print(torch.where(cond>0.5,a,b))

#gather
prob = torch.rand(4,10)
idx = prob.topk(k = 3,dim = 1)
print(idx)
idx = idx[1]
print(idx)
lable = torch.arange(10)+100
print(lable)
print(torch.gather(lable.expand(4,10),dim = 1,index = idx.long()))

