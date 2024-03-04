from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # 读取图像
    image = Image.open(image_path)

    # 转换为灰度图
    gray_image = image.convert("L")

    # 将PIL图像转换为OpenCV图像格式
    gray_image_cv = np.array(gray_image)

    # 降噪：使用高斯模糊
    blurred_image = cv2.GaussianBlur(gray_image_cv, (5, 5), 0)

    return image, gray_image, blurred_image

# 使用函数
original_image, gray_image, processed_image = process_image('/Users/zhengjiatong/Desktop/Fig01.tif')

# 应用Canny边缘检测
canny_edges = cv2.Canny(processed_image, 120, 200)

# 轮廓检测
contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制轮廓
contoured_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)  # 将灰度图转换为BGR颜色图以便彩色绘制
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)  # 用绿色绘制轮廓

# 米粒数量
rice_grains_count = len(contours)

# 展示结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(contoured_image)
plt.title('Contours on Original Image')
plt.axis('off')

plt.show()

print('quantity:',rice_grains_count)