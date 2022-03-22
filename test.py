from model_plus import NewNet
from PIL import Image
import torch
from torchvision.transforms import Compose, Grayscale, ToTensor
import matplotlib.pyplot as plt
import cv2
import numpy as np


# 设定阈值与类别
THRESHOLD_VALUE = 0.01
class_number = "Class6"

# 指定模型
model = NewNet()
# 载入模型
model.load_state_dict(torch.load(f'./model/NewNet_model_{class_number}.pt'))
# 使模型进入测试模式
model.eval()
# 指定使用GPU
model = model.to('cuda')

# 1、图片路径
imgpath = f"./image/{class_number}/img2.PNG"
# 以灰度图读取图片
img = Image.open(imgpath).convert('L')
# 定义数据预处理操作：将图像转化为灰度图像，将其转化为tensor格式
transform = Compose([Grayscale(), ToTensor()])
# 使用预处理的方式读取图片
img = transform(img)
# 以GPU的方式运行
img = img.to('cuda')
# 对数据0维度进行扩充
img = img.unsqueeze(0)
# numpy格式的原图像
x = img.detach().cpu().numpy()[0][0]

# 2、模型输出结果为y
y = model(img)
# numpy格式的预测图像
EDx = y[0][0].detach().cpu().numpy()

# 3、进行相减，结果保留绝对值
S = torch.abs(img[0][0]-y[0][0])
# numpy格式的EDX
s_EDX = S.detach().cpu().numpy()

# 4、S操作
MAX_VALUE = 255
S_threshold = S.detach().cpu().numpy()
ret, thresh_basic = cv2.threshold(S_threshold, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)

# 5、膨胀
kernel = np.ones((3, 3), np.uint8)
img_erosion = cv2.dilate(thresh_basic, kernel, iterations=1)

# 6、去除椒盐噪声
img_median = cv2.medianBlur(img_erosion, 5)

# 7、Canny边缘检测
THRESHOLD1 = 1
THRESHOLD2 = 0

edged = cv2.Canny((img_median*255).astype(np.uint8),THRESHOLD1,THRESHOLD2)

img_original = img.detach().cpu().numpy()[0][0]
img_3 = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
img_3 = np.power(img_3, 3)

# 8:轮廓
thresh_inv_uint8 = img_median.astype(np.uint8)
contours, h = cv2.findContours(thresh_inv_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

num_of_con = str(len(contours) - 1)
imageOri = img_3
CON_COLOR = (255, 0, 0)
CON_THICKNESS = 3

highlighted_img = cv2.drawContours(imageOri, contours, -1, CON_COLOR, CON_THICKNESS)

# 总图
plt.figure(figsize=(15,10))

plt.subplot(241)
plt.imshow(img.detach().cpu().numpy()[0][0],cmap=plt.cm.gray)
plt.title('X')
plt.axis('off')

plt.subplot(242)
plt.imshow(y[0][0].detach().cpu().numpy(),cmap=plt.cm.gray)
plt.title('ED(X)')
plt.axis('off')

plt.subplot(243)
plt.imshow(S.detach().cpu().numpy(),cmap=plt.cm.gray)
plt.title('S')
plt.axis('off')

plt.subplot(244)
plt.imshow(thresh_basic,cmap=plt.cm.gray)
plt.title(f'thresholded={THRESHOLD_VALUE}')
plt.axis('off')

plt.subplot(245)
plt.imshow(img_erosion,cmap=plt.cm.gray)
plt.title('Dilate')
plt.axis('off')

plt.subplot(246)
plt.imshow(img_median,cmap=plt.cm.gray)
plt.title('Filter')
plt.axis('off')

plt.subplot(247)
plt.imshow(edged,cmap=plt.cm.gray)
plt.title('Edge Detection')
plt.axis('off')

plt.subplot(248)
plt.imshow(highlighted_img, cmap=plt.cm.gray)
plt.title('Result')
plt.axis('off')

plt.tight_layout()

plt.show()
