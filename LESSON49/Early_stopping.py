#Tricks
'''
Early stopping
dropout
stochastic gradient descent
'''

#es
'''
validation set to select parameters
monitor validation performance
stop at the highest val perf:这是一个经验值
'''

#do
'''
learning less to learn better
each connection may lose
each connection has p=[0,1] to lose
'''

'''
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(784,200),
    torch.nn.Dropout(0.5),#drop 50% of the neuron,本来是全部直连，使用dropout会断掉一些
    torch.nn.ReLU(),
    torch.nn.Linear(200,200),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nnLinear(200,10),
)
'''

#behavior between train and test
#dropout 不能再test，只有train部分有dropout
'''
for epoch in range(epochs):
    net_dropped.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        ...
        net_dropped.eval()#取消dropout
        test_loss = 0
        correct = 0
        for data,target in test_loader:
            ...
'''

#stochastic gradient descent：随机梯度下降
'''
与stochastic对应的是deterministic
'''