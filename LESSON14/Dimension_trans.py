import torch
'''Operation:'''
'''1. view|reshape'''
'''2. squeeze|unsqueeze'''
'''3. transpose|.t|permute'''
'''4. expand|repeat'''

a = torch.rand(4,1,28,28)
print(a.view(4,1*28*28))
print(a.view(4,1*28*28).shape)
'''logic error'''
b = a.view(4,1*28*28)
print(b.view(4,28,28,1))
print(b.view(4,28,28,1).shape)


'''squeeze|unsqueeze'''
'''Notice spectrum[-5,5)'''
print(a.shape)
print(a.unsqueeze(0).shape)
print(a.unsqueeze(-1).shape)

a = torch.tensor([1.2,2.3])
print(a)
print(a.shape)
print(a.unsqueeze(-1))

'''f + b'''
b = torch.rand(32)
f = torch.rand(4,32,14,14)
b = b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
print(b.shape)

print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(-1).shape)


'''expand|repeat'''
'''expand|broadcasting'''
'''repeat|memortcopied'''
a = torch.rand(4,32,14,14)
print(b.expand(4,32,14,14))
print(b.shape)
print(b.expand(-1,32,-1,-1).shape)
print(b.expand(-1,32,-1,-4).shape)

print(b.repeat(4,32,1,1))
print(b.repeat(4,1,1,1))
print(b.repeat(4,1,32,32))


'''transpose|.t|permute'''
a = torch.randn(3,4)
print(a)
print(a.t())

a = torch.randn(4,3,32,32)
a1 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,3,32,32)
a2 = a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)
print(a.transpose(1,3).shape)
print(torch.all(torch.eq(a,a1)))
print(torch.all(torch.eq(a,a2)))

'''permute'''
a = torch.rand(4,3,28,28)
print(a.transpose(1,3).shape)
b = torch.rand(4,3,28,32)
print(b.transpose(1,3).shape)
print(b.transpose(1,3).transpose(1,2).shape)
print(b.permute(0,2,3,1).shape)