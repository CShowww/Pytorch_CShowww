import torch
'''cat'''
'''stack'''
'''split'''
'''chunk'''

a = torch.rand(4,32,8)
b = torch.rand(5,32,8)
print(torch.cat([a,b],dim = 0).shape)

a1 = torch.rand(4,3,32,32)
a2 = torch.rand(5,3,32,32)
print(torch.cat([a1,a2],dim = 0).shape)

a2 = torch.rand(4,1,32,32)
print(torch.cat([a1,a2],dim = 1).shape)








'''stack'''

a1 = torch.rand(4,3,16,32)
a2 = torch.rand(4,3,16,32)
print(torch.stack([a1,a2],dim = 1).shape)

a = torch.rand(32,8)
b = torch.rand(32,8)
print(torch.stack([a,b],dim = 0).shape)


'''split'''
a = torch.rand(32,8)
b = torch.rand(32,8)
c = torch.stack([a,b],dim = 0)
print(c.shape)

aa,bb = c.split([1,1],dim = 0)
print(aa.shape,bb.shape)
aa,bb = c.split(1,dim = 0)
print(aa.shape,bb.shape)
d = torch.rand(32,8)
c = torch.stack([a,b,d],dim = 0)
aa,bb = c.split([1,2],dim = 0)
print(aa.shape,bb.shape)



'''chunk'''
a = torch.rand(32,8)
b = torch.rand(32,8)
c = torch.stack([a,b],dim = 0)
print(c.shape)

aa,bb = c.chunk(2,dim = 0)
print(aa.shape,bb.shape)