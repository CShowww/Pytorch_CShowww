import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision

x = torch.randn(1,784)
layer1 = nn.Linear(784,200)
layer2 = nn.Linear(200,200)
layer3 = nn.Linear(200,10)

x = layer1(x)
print(x.shape)
x = layer2(x)
print(x.shape)
x = layer3(x)
print(x.shape)


#relu
x = torch.randn(1,784)
x = layer1(x)
x = F.relu(x,inplace=True)
x = layer3(x)
x = F.relu(x,inplace=True)

#example
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,10),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.model(x)
        return x


#Train
epochs = 3
learning_rate = 1e-2
batch_size = 64

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('datasets/mnist_data',
                train=True,
                download=True,
                transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),                       # 数据类型转化
                torchvision.transforms.Normalize((0.1307, ), (0.3081, )) # 数据归一化处理
    ])), batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('datasets/mnist_data/',
                train=False,
                download=True,
                transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])),batch_size=batch_size,shuffle=False)

net = MLP()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))