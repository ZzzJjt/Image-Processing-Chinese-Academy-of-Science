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
original_image, gray_image, processed_image = process_image('/Users/zhengjiatong/Desktop/Fig01.tif')  # 替换为您的图像路径

# 展示结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Image')
plt.axis('off')

plt.show()

# 预处理：使用高斯模糊进行降噪
blurred = cv2.GaussianBlur(processed_image, (5, 5), 0)

# 二值化处理
# 我们可以先尝试自动阈值方法
_, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 轮廓检测
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制轮廓
contoured_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)  # 将灰度图转换为BGR颜色图以便彩色绘制
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)  # 用绿色绘制轮廓

# 显示二值化图像和轮廓检测结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(contoured_image)
plt.title('Contours on Original Image')
plt.axis('off')

plt.show()

# 米粒数量
rice_grains_count = len(contours)
print(f"Detected Rice Grains: {rice_grains_count}")


# 重新调整图像处理参数，确保考虑整个图像

# 调整高斯模糊参数
blurred_adjusted = cv2.GaussianBlur(binary_image, (3, 3), 0)

# 尝试不同的阈值以更好地捕捉图像中心的米粒
_, binary_adjusted = cv2.threshold(blurred_adjusted, 100, 255, cv2.THRESH_BINARY_INV)

# 应用形态学操作以改善米粒的分离
kernel_adjusted = np.ones((2, 2), np.uint8)
eroded_adjusted = cv2.erode(binary_adjusted, kernel_adjusted, iterations=1)
dilated_adjusted = cv2.dilate(eroded_adjusted, kernel_adjusted, iterations=1)

# 再次进行轮廓检测，确保考虑整个图像
contours_adjusted, _ = cv2.findContours(dilated_adjusted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制轮廓
contoured_img_adjusted = cv2.cvtColor(blurred_adjusted, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contoured_img_adjusted, contours_adjusted, -1, (0, 255, 0), 1)

# 计算米粒数量
rice_grains_count_final = len(contours_adjusted)

# 展示调整后的结果
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(binary_adjusted, cmap='gray')
plt.title('Binary Image Adjusted')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(dilated_adjusted, cmap='gray')
plt.title('Dilated Image Adjusted')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(contoured_img_adjusted, cv2.COLOR_BGR2RGB))
plt.title('Contours on Adjusted Image')
plt.axis('off')

plt.show()

print('quantity:',rice_grains_count_final)


