import torch

'''a:[batch,channel,length,width'''

a = torch.rand(4,3,28,28)
print(a[0].shape)
print(a[0,0].shape)
print(a[0,0,2,4])

'''select first|last N'''

print(a[:2].shape)
print(a[:2,:1,:,:].shape)
print(a[:2,1:,:,:].shape)
print(a[:2,:-1,:,:].shape)

'''select by step'''
print(a[:,:,0:28:2,0:28:2].shape)
print(a[:,:,::2,::2].shape)


'''select by specific index'''
print(a.index_select(0,torch.tensor([0,2])).shape)
print(a.index_select(1,torch.tensor([1,2])).shape)
print(a.index_select(2,torch.arange(28)).shape)
print(a.index_select(2,torch.arange(8)).shape)


'''special'''
print(a[...].shape)
print(a[0,...].shape)
print(a[:,1,...].shape)
print(a[...,:2].shape)


'''select by mask'''
x = torch.randn(3,4)
print(x)
mask = x.ge(0.5)
print(torch.masked_select(x,mask))


'''select by flatten index'''
src = torch.tensor([[4,3,5],[6,7,8]])
print(torch.take(src,torch.tensor([0,2,-1])))
