from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.filters.rank import median
from skimage.morphology import disk
import pywt
image1 = Image.open("/Users/zhengjiatong/Desktop/Fig01.tif")
image1_array = np.array(image1)
original_array = image1_array
image2 = Image.open("/Users/zhengjiatong/Desktop/Fig02.tif")
image2_array = np.array(image2)
image3 = Image.open("/Users/zhengjiatong/Desktop/Fig03.tif")
image3_array = np.array(image3)
image4 = Image.open("/Users/zhengjiatong/Desktop/Fig04.tif")
image4_array = np.array(image4)
noisy_arrays = [image1_array, image2_array, image3_array,image4_array]  # 这里的 image1, image2, image3 应该被替换为实际的图像数组

mse_values = []
snr_values = []
def wavelet_denoise(image, wavelet='db1', level=2):
    # 小波分解
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # 计算阈值
    threshold = (np.median(np.abs(coeffs[-level])) / 0.6745) * (2*np.log(image.size))**0.5
    # 对每一层的小波系数进行阈值处理
    new_coeffs = list(map(lambda coeff_tuple: tuple(pywt.threshold(c, threshold, mode='soft') for c in coeff_tuple), coeffs))
    # 小波重构
    return pywt.waverec2(new_coeffs, wavelet)
def mse(imageA, imageB):
    """计算两个图像之间的均方误差"""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def snr(image, noise):
    """计算信噪比"""
    signal_power = np.sum(image.astype("float") ** 2)
    noise_power = np.sum(noise.astype("float") ** 2)
    return 10 * np.log10(signal_power / noise_power)
# # 应用小波去噪
# denoised_images = [wavelet_denoise(image) for image in noisy_arrays]
# 应用小波去噪并计算 MSE 和 SNR
for noisy in noisy_arrays:
    denoised = wavelet_denoise(noisy)

    # 计算 MSE 和 SNR
    mse_value = mse(original_array, denoised)
    snr_value = snr(original_array, original_array - denoised)

    mse_values.append(mse_value)
    snr_values.append(snr_value)

# 展示去噪后的图像和度量值
plt.figure(figsize=(12, 18))
for i, denoised in enumerate(denoised_images, start=1):
    ax = plt.subplot(4, 2, i)
    plt.imshow(denoised, cmap='gray')
    plt.title(f'Denoised Image {i}')
    plt.axis('off')
    # 在图像旁添加 MSE 和 SNR 值
    plt.text(0.5, -0.1, f'MSE: {mse_values[i-1]:.2f}, SNR: {snr_values[i-1]:.2f}',
             ha='center', va='top', transform=ax.transAxes)

plt.tight_layout()
plt.show()

# 打印每个图像的 MSE 和 SNR 值
print("MSE Values:", mse_values)
print("SNR Values:", snr_values)


