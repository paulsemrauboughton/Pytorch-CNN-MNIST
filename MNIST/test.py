import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(12, 28, 5)
        self.fc1   = nn.Linear(28 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

net = NeuralNet()
net.load_state_dict(torch.load('trained_net.pth', map_location=torch.device('cpu')))
net.eval()

img = Image.open('MNIST/image.png')
test_image = transform(img)
test_image = 1 - test_image

with torch.no_grad():
    test_image = test_image.unsqueeze(0)  # shape: [1, 1, 28, 28]
    output = net(test_image)
    probabilities = F.softmax(output, dim=1)

_, predicted_label = torch.max(output, 1)
predicted_probability = probabilities[0, predicted_label].item()

plt.imshow(test_image.squeeze(), cmap="gray")
plt.title(f"Predicted: {predicted_label.item()} ({predicted_probability:.2f})")
plt.show()