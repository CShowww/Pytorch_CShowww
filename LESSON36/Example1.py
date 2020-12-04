import torch
import numpy as np
import matplotlib.pyplot as plt

#f(x,y) =
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
print('x,y range:',x.shape,y.shape)
X,Y = np.meshgrid(x,y)
print('X,Y maps:',X.shape,Y.shape)
Z = himmelblau([X,Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection = '3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x = torch.tensor([0.,0.],requires_grad=True)
optimizer = torch.optim.Adam([x],lr = 1e-3)
for step in range(20002):
    pred = himmelblau(x)
    optimizer.zero_grad()
    pred.backward()

    optimizer.step()
    if step%2000==0:
        print(f"step {step}: x = {x.tolist()} f(x) = {pred.item()}")