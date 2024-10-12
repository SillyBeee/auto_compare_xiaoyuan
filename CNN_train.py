import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import os
import torch.nn.functional as F
# 定义超参数
batch_size = 64 # 批量大小
learning_rate = 0.001   #学习率
num_epochs = 10  #训练轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 使用GPU加速

print(f'Using device: {device}')
# 数据预处理
transform = transforms.Compose([ # 图像预处理
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
    transforms.Normalize((0.1307,), (0.3081,))
])
class Mydatasets(Dataset):
    def __init__(self, root_dir, transform=None):  # 构造函数
        self.root_dir = root_dir #根目录路径
        self.transform = transform  #预处理步骤
        self.images = []   #空列表，用于存储每一张图片路径
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            self.images.extend([(os.path.join(label_dir, img), int(label)) for img in os.listdir(label_dir)])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name , label = self.images[idx]
        label = int(os.path.basename(os.path.dirname(img_name)))
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        if self.transform:#如果有预处理步骤，则预处理
            image = self.transform(image)
        return image, label
    

# 加载数据集
train_dataset = Mydatasets(root_dir = 'train_datasets' , transform = transform )
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# 定义简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleNet()
model.to(device)
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器

# 训练模型
for epoch in range(num_epochs):  
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
#保存模型
torch.save(model.state_dict(), 'model.pth')
print("Model weights saved to model.pth")