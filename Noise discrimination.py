import cv2
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def analyze_and_estimate_noise(image, title):
    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算统计特征
    mean = np.mean(gray_image)
    variance = np.var(gray_image)
    skewness = stats.skew(gray_image.flatten())
    kurtosis = stats.kurtosis(gray_image.flatten())

    # 估计噪声类型
    noise_type = "Unknown"
    if variance > 5000:
        if -0.5 < skewness < 0.5 and -0.5 < kurtosis < 0.5:
            noise_type = "Gaussian Noise"
        elif skewness > 0.5 or skewness < -0.5:
            noise_type = "Gamma or Exponential Noise"
        elif kurtosis > 1 or kurtosis < -1:
            noise_type = "Salt-and-Pepper Noise"
    else:
        noise_type = "可能无明显噪声或噪声水平较低"

    return gray_image, f"{title}\nNoise Type: {noise_type}"

# 图像路径
image_paths = [
    '/Users/zhengjiatong/Desktop/Fig02.tif',
    '/Users/zhengjiatong/Desktop/Fig03.tif',
    '/Users/zhengjiatong/Desktop/Fig04.tif'
]

# 创建子图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 分析每张图像并绘制直方图
for idx, path in enumerate(image_paths):
    try:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {path}")

        gray_image, title = analyze_and_estimate_noise(image, path.split('/')[-1])
        axs[idx].hist(gray_image.flatten(), bins=50, color='blue', alpha=0.7)
        axs[idx].set_title(title)
        axs[idx].set_xlabel("Pixel values")
        axs[idx].set_ylabel("Frequency")
    except Exception as e:
        print(f"错误: {e}")

plt.tight_layout()
plt.show()
