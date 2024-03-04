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
# Function to apply Gaussian and median filtering
def wavelet_denoise(image, wavelet='db1', level=1):
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

def apply_filters(noisy_image):
    # Apply Gaussian filter
    gaussian_filtered = gaussian(noisy_image, sigma=4)

    # Apply median filter
    median_filtered = median(noisy_image, disk(4))

    return gaussian_filtered, median_filtered

# Apply filters and calculate MSE and SNR for each filtered image
filtered_images = []
for noisy in noisy_arrays:
    gaussian_filtered, median_filtered = apply_filters(noisy)
    filtered_images.append((gaussian_filtered, median_filtered))

    # Calculating MSE and SNR for Gaussian filtered image
    mse_gaussian = mse(original_array, gaussian_filtered)
    snr_gaussian = snr(original_array, original_array - gaussian_filtered)

    # Calculating MSE and SNR for Median filtered image
    mse_median = mse(original_array, median_filtered)
    snr_median = snr(original_array, original_array - median_filtered)

    mse_values.append((mse_gaussian, mse_median))
    snr_values.append((snr_gaussian, snr_median))

# Display the filtered images
plt.figure(figsize=(12, 18))  # 调整图形大小以适应更多子图
for i, (gaussian_img, median_img) in enumerate(filtered_images, start=1):
    # 对于高斯滤波后的图像
    ax1 = plt.subplot(4, 2, 2*i-1)
    plt.imshow(gaussian_img, cmap='gray')
    plt.title(f'Gaussian Filtered Image {i}')
    plt.axis('off')
    # 应用小波去噪
    denoised_images = [wavelet_denoise(image) for image in noisy_arrays]
    # 在图像旁添加 MSE 和 SNR 值
    plt.text(0.5, -0.1, f'MSE: {mse_values[i-1][0]:.2f}, SNR: {snr_values[i-1][0]:.2f}',
             ha='center', va='top', transform=ax1.transAxes)

    # 对于中值滤波后的图像
    ax2 = plt.subplot(4, 2, 2*i)
    plt.imshow(median_img, cmap='gray')
    plt.title(f'Median Filtered Image {i}')
    plt.axis('off')
    # 在图像旁添加 MSE 和 SNR 值
    plt.text(0.5, -0.1, f'MSE: {mse_values[i-1][1]:.2f}, SNR: {snr_values[i-1][1]:.2f}',
             ha='center', va='top', transform=ax2.transAxes)

plt.tight_layout()
plt.show()



print(mse_values, snr_values)

