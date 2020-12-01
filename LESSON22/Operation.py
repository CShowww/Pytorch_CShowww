'''
basic = Add/minus/multiply/divide
matmul
pow
sqrt/rsqrt
round
'''
import torch

#广播机制
a = torch.rand(3,4)
b = torch.rand(4)

print(a+b)
print(torch.add(a,b))

print(torch.all(torch.eq(a-b,torch.sub(a,b))))
print(torch.all(torch.eq(a*b,torch.mul(a,b))))
print(torch.all(torch.eq(a/b,torch.div(a,b))))

'''matmul'''
a = torch.ones(2,2)*3
b = torch.ones(2,2)
print(torch.mm(a,b))
print(torch.matmul(a,b))
print(a@b)

'''eg'''
w = torch.rand(512,784)
x = torch.rand(4,784)
print((x@w.t()).shape)

a = torch.rand(4,3,28,64)
b = torch.rand(4,3,64,32)
#print(torch.mm(a,b).shape)
print(torch.matmul(a,b).shape)

b = torch.rand(4,1,64,32)
print(torch.matmul(a,b).shape)

'''pow'''
a = torch.full([2,2],3)
print(a.pow(2))
print(a**2)
aa = a**2

'''sqrt/rsqrt'''
print(aa.sqrt())
print(aa.rsqrt())

'''exp/log'''
a = torch.exp(torch.ones(2,2))
print(a)
print(torch.log(a))


'''approximation
floor(),ceil()
round()
trunc(),frac()
'''

a = torch.tensor(3.14)
print(a.floor())
print(a.ceil())
print(a.trunc())
print(a.frac())
print(a.round())


'''clamp'''
grad = torch.rand(2,3)*15
print(grad)
print(grad.max())
print(grad.median())
print(grad.clamp(10))
print(grad.clamp(0,10))