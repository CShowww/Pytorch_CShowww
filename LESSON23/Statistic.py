import torch
'''
norm
mean sum
prod
max/min/argmin/argmax
kthvalue
topk
'''

a = torch.full([8],1)
b = a.view(2,4)
c = a.view(2,2,2)

print(a.norm(1),b.norm(1),c.norm(1))
print(a.norm(2),b.norm(2),c.norm(2))
print(b.norm(1,dim = 1))
print(b.norm(2,dim = 1))

print(c.norm(1,dim = 0))
print(c.norm(2,dim = 0))



'''meadn/sum/min/max/prod'''
a = torch.arange(8).view(2,4).float()
print(a)
print(a.min(),a.max(),a.mean(),a.prod(),a.sum())

print(a.argmax(),a.argmin())

a = a.view(1,2,4)
print(a.argmax())
print(a.argmin())

a = torch.rand(2,3,4)
print(a.argmax())

a = torch.randn(4,10)
print(a[0])
print(a.argmax())
print(a.argmax(dim = 1))


'''dim/keepdim'''

a  = torch.rand(4,10)
print(a.max(dim = 1))
print(a.argmax(dim = 1))
print(a.max(dim = 1,keepdim = True))
print(a.argmax(dim = 1,keepdim=True))

'''top-k/k-th'''
a = torch.rand(4,10)
print(a.topk(3,dim = 1))
print(a.topk(3,dim = 1,largest=False))

#kthvalue,第几小
print(a.kthvalue(8,dim=1,keepdim=True))


'''compare'''
a = torch.rand(4,10)
print(a)
print(a>0)
print(a!=0)

a = torch.ones(2,3)
b = torch.randn(2,3)
print(torch.eq(a,b))
print(torch.eq(a,a))

