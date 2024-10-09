import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 定义模型架构
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将输入展平成一维向量
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载预训练的模型权重
def load_model(weights_path):
    model = SimpleNet()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

# 图像预处理和分割
def preprocess_and_segment(image):
    # 读取图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # 轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit = image[y:y+h, x:x+w]
        
        # 将数字图像调整为 28x28 大小
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit = np.expand_dims(digit, axis=-1)
        digit = digit.astype(np.float32) / 255.0
        digits.append(digit)
    
    return digits

# 识别多个数字
def recognize_digits(image, model):
    digits = preprocess_and_segment(image)
    if not digits:
        return "No digits found"
    
    # 将所有数字图像堆叠成一个批次
    digits = np.array(digits)
    digits = np.expand_dims(digits, axis=1)  # 添加通道维度
    digits = torch.from_numpy(digits)
    
    model.eval()
    with torch.no_grad():
        outputs = model(digits)
        _, predicted = torch.max(outputs.data, 1)
    
    # 组合结果
    result = ''.join(map(str, predicted.numpy()))
    return result

# 封装整个流程的函数
def recognize_digits_from_image(image_path, weights_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 加载模型
    model = load_model(weights_path)
    
    # 识别数字
    result = recognize_digits(image, model)
    return result

# 主函数
if __name__ == "__main__":
    # 模型权重路径
    weights_path = 'model/model.pth'
    
    # 图像路径
    image_path = 'path/to/your/image.png'
    
    # 识别数字
    # result = recognize_digits_from_image(image_path, weights_path)
    # print(f'Recognized digits: {result}')