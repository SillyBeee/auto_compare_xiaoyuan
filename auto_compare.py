import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from time import sleep
import os
print(torch.cuda.is_available()) #输出是否支持cuda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 如果支持cuda，则使用cuda
print(f'Using device: {device}')

transform = transforms.Compose([# 图像预处理
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 定义模型架构
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

# 加载预训练的模型权重
def load_model(weights_path):
    model = SimpleNet()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

# 图像预处理，返回二值化图像
def preprocess(image):
    # 读取图像
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # binary=255-binary
    return binary
    
    # cv2.imshow("image", binary)
    # cv2.waitKey(0)

#剪切ROI区域
def crop_pics(image,crop_areas):
    # cv2.imshow("image", image)
    cropped_img_list=[]
    for i, crop_area in enumerate(crop_areas, start=1):
        x1, y1, x2, y2 = crop_area
        cropped_image =image[y1:y2, x1:x2]
        cropped_img_list.append(cropped_image)
    if len(cropped_img_list)!=2:
        print("cropped length error")
        exit()
        
    return  cropped_img_list
def show_tensor(image):
    image=transforms.ToPILImage()(image)
    image.show()
    cv2.waitKey(0)
#推理，输入图片，输出预测
def detect(image):
    image=255-image
    if image is None:
        print("供推理图片为空")
        exit()
    with torch.no_grad():
            tensor=cv2.resize(image,(28,28))
            image_tensor = transform(tensor)
            # show_tensor(image_tensor)
            image_tensor=image_tensor.to(device)
            #mat转为tensor
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            print(f'Predicted digit: {predicted.item()}')
            return int(predicted.item())
            

def recognize_digits(image, rgb_image):
    contours, _=cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = rgb_image.copy()
    cnt=0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(contour_img, [contour], 0, (0, 255-100*cnt, 0), 2)
        cv2.rectangle(contour_img, (x-25, y-20), (x + w+25, y + h+20), (0, 255, 0), 2)
        cv2.putText(contour_img, str(cnt), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cnt+=1
    # cv2.imshow("contour_img", contour_img)
    # cv2.waitKey(0)
    num = []
    
    # cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    if len(contours) ==1:
        x, y, w, h = cv2.boundingRect(contours[0])
        if w > 10 and h > 10:
                digit = image[y-20:y+h+20, x-25:x+w+25]
                return detect(digit)
    if len(contours) == 2:
        min_x=1000
        count=0
        target=-1
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if  w > 10 and h > 10: #当找到符合条件且最左的数字时，记录其位置             
                digit = image[y-20:y+h+20, x-25:x+w+25]
                num_predict = detect(digit)
                num.append(num_predict)
            if x<min_x:
                min_x=x
                target=count
            count+=1
        if target == 0:
            return num[0]*10+num[1]
        if target == 1:
            return num[0]+num[1]*10

def take_screen_shot(screen_shot_path):
    
    os.system(f'adb shell screencap -p > {screen_shot_path}')
 
    # 读取截取的屏幕截图并替换行结束符
    with open(screen_shot_path, 'rb') as f:
        return f.read().replace(b'\r\n', b'\n')
      
def draw_bigger():
    os.system("adb shell input swipe 450 1800 850 1900 1")
    os.system("adb shell input swipe 850 1900 450 2000 1")

def draw_smaller():
    os.system("adb shell input swipe 850 1800 450 1900 1")
    os.system("adb shell input swipe 450 1900 850 2000 1")

def compare_numbers(left, right):
    
    try:
        if left > right:
            print(f"{left} > {right}")
            draw_bigger()
        else:
            print(f"{left} < {right}")
            draw_smaller()
    except ValueError:
        print("数字格式无效。")

# 主函数
if __name__ == "__main__":
    #图片裁剪区域，四个参数为左，上，右，下
    crop_areas = [
        (300, 720, 530, 880),
        (700, 720, 930, 880)
    ]
    

    # 模型权重路径
    weights_path = 'model/model.pth'
    #快照路径
    screen_shot_path = 'screen_shot/screenshot.png'
    # 图像路径
    image_path = 'screen_shot/screenshot.png'
    model=SimpleNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    while True:
        take_screen_shot(screen_shot_path)
        img=cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print("读取图片失败")
            continue
        grey=preprocess(img) 
        img_list=crop_pics(grey,crop_areas)#进行扣图
        rgb_img_list=crop_pics(img,crop_areas)
        left_num=recognize_digits(img_list[0],rgb_img_list[0] )
        right_num=recognize_digits(img_list[1], rgb_img_list[1] )
        if (left_num is None) or (right_num is None):
            print("未识别到数字")
            continue
        compare_numbers(left_num, right_num)
        sleep(0.3)
    # img=cv2.imread(image_path, cv2.IMREAD_COLOR)
    # grey=preprocess(img) 
    # img_list=crop_pics(grey,crop_areas)#进行扣图
    # rgb_img_list=crop_pics(img,crop_areas)
    # left_num=recognize_digits(img_list[0],rgb_img_list[0] )
    # right_num=recognize_digits(img_list[1], rgb_img_list[1] )
    
    # cv2.imshow("img", img_list[0])
    # cv2.imshow("img2",img_list[1]) 
    # cv2.waitKey(0)
    
    
    