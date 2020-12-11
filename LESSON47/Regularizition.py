#如果没有必要，就不要加
'''
reduce overfitting
1.more data
2. constrain model complexity{shallow,regularization}
3.dropout
4.data argumentation
5.early stopping
'''
#regularization也叫 weight decay
'''
加上一个一范数累加
迫使范数接近于0，从而减少模型复杂度
'''

#L1-regularization:需要自己实现，是一范数
'''
regulatization_loss = 0
for param in model.parameters():
    regulatization_loss += torch.sum(torch.abs(param))
classify_loss = criteon(logits,target)
loss = classify_loss + 0.01 * regulatization_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()
'''

#L2-regularization:weight——decay，直接在优化器中实现
'''
device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(),lr = learning_rate,weight_decay = 0.01)
criteon = nn.CrossEntropyLoss().to(device)
'''

#如果没有overfitting，使用weight_decay会约束性能，从而降低perfromance
