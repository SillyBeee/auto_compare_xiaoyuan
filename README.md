# 这是一个基于OpenCV和PyTorch的小猿口算自动比大小程序

## 环境配置🖥
```
- Python 3.8
- OpenCV 4.5.3
- PyTorch 1.10.0
- Torch 2.4.1 + Cuda 11.8  
- Torchvision 0.11.1
- adb 1.0.41
```
## 目录结构 📂
```
- CNN_train.py    用于训练卷积神经网络的程序
- auto_compare.py      主程序
- model           训练好的模型权重
- screen_shot      用于存储手机截图和读取手机截图

```
## 程序运行 🚀
安装完成opencv及pytorch全家桶后，安卓手机开启开发者模式并开启USB调试，将手机连接到电脑，开启20以内比大小对决，运行auto_compare.py即可
 
## 程序可能有些许bug，请多多包涵🙏
