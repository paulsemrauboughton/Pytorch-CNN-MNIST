import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),  #Convert PIL image to Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

subset_size = int(len(train_dataset) * 0.5)  # % of the data used for computational efficiency.
indices = np.random.choice(len(train_dataset), subset_size, replace=False)
small_train_dataset = torch.utils.data.Subset(train_dataset, indices)

train_loader =  torch.utils.data.DataLoader(small_train_dataset, batch_size=32, shuffle=True)

image, label = train_dataset[0]
print(image.size())
torch.Size([1,28,28])

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 12, 5)  # 28 - 5 = 23 / 1 = 23, 23+1 = 24  (12,24,24)
        self.pool = nn.MaxPool2d(2,2)   #  (12,12,12)
        self.conv2 = nn.Conv2d(12, 28, 5)   # 12 - 5 = 7, 7+1 = 8   (28, 8, 8) -> (28,5,5) -> (28*5*5)
        self.fc1 = nn.Linear(28 * 4 * 4, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range (30):
    print (f'Training Epoch {epoch}... ')
    
    running_loss =  0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print (f'Loss: {running_loss / len(train_loader): .4f}')


torch.save(net.state_dict(), 'trained_net.pth')
