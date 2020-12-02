import torch
x = torch.tensor(1.)
w1 = torch.tensor(2.,requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2.,requires_grad=True)
b2 = torch.tensor(1.)
y1 = x*w1 + b1
y2 = y1*w2 + b2

dy2_dy1 = torch.autograd.grad(y2,[y1],retain_graph=True)
dy1_dw1 = torch.autograd.grad(y1,[w1],retain_graph=True)
dy2_dw1 = torch.autograd.grad(y2,[w1],retain_graph=True)

print(dy2_dw1[0].equal(dy2_dy1[0]*dy1_dw1[0]))