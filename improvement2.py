from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
def nl_means_denoise(image):
    # 将图像转换为浮点型，非局部均值函数在浮点图像上效果最佳
    float_image = img_as_float(image)
    # 估计图像的噪声标准差
    sigma_est = np.mean(estimate_sigma(float_image))
    # 应用非局部均值去噪
    denoised_img = denoise_nl_means(float_image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
    return denoised_img
def calculate_mse(image_a, image_b):
    """计算两幅图像之间的MSE"""
    return np.mean((image_a - image_b) ** 2)

def calculate_snr(original_image, noise_image):
    """计算信噪比"""
    signal_power = np.mean(original_image ** 2)
    noise_power = np.mean((original_image - noise_image) ** 2)
    return 10 * np.log10(signal_power / noise_power)
def display_image_with_metrics(ax, image, mse, snr, title):
    ax.imshow(image, cmap='gray')
    ax.set_title(f'{title}\nMSE: {mse:.2f}, SNR: {snr:.2f} dB')
    ax.axis('off')
# Function to resize images to a common shape
def resize_image(image, reference_shape):
    return resize(image, reference_shape, anti_aliasing=True)
# 加载图像并应用非局部均值去噪
image1 = io.imread('/Users/zhengjiatong/Desktop/Fig01.tif')  # 替换为您的图像路径
denoised_image1 = nl_means_denoise(image1)
image2 = io.imread('/Users/zhengjiatong/Desktop/Fig02.tif')  # 替换为您的图像路径
denoised_image2 = nl_means_denoise(image2)
image3 = io.imread('/Users/zhengjiatong/Desktop/Fig03.tif')  # 替换为您的图像路径
denoised_image3 = nl_means_denoise(image3)
image4 = io.imread('/Users/zhengjiatong/Desktop/Fig04.tif')  # 替换为您的图像路径
denoised_image4 = nl_means_denoise(image4)

# Resize the denoised images to match the original images' sizes
denoised_image1_resized = resize_image(denoised_image1, image1.shape)
denoised_image2_resized = resize_image(denoised_image2, image2.shape)
denoised_image3_resized = resize_image(denoised_image3, image3.shape)
denoised_image4_resized = resize_image(denoised_image4, image4.shape)

# Recalculate MSE and SNR
mse1 = calculate_mse(image1, denoised_image1_resized)
snr1 = calculate_snr(image1, denoised_image1_resized)
mse2 = calculate_mse(image2, denoised_image2_resized)
snr2 = calculate_snr(image2, denoised_image2_resized)
mse3 = calculate_mse(image3, denoised_image3_resized)
snr3 = calculate_snr(image3, denoised_image3_resized)
mse4 = calculate_mse(image4, denoised_image4_resized)
snr4 = calculate_snr(image4, denoised_image4_resized)

# Plot the denoised images with MSE and SNR values
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

display_image_with_metrics(axs[0, 0], denoised_image1_resized, mse1, snr1, 'Image 1')
display_image_with_metrics(axs[0, 1], denoised_image2_resized, mse2, snr2, 'Image 2')
display_image_with_metrics(axs[1, 0], denoised_image3_resized, mse3, snr3, 'Image 3')
display_image_with_metrics(axs[1, 1], denoised_image4_resized, mse4, snr4, 'Image 4')

plt.tight_layout()
plt.show()
# Plot the denoised images with MSE and SNR values

# Check if the images are loaded correctly by plotting them
# fig, axs = plt.subplots(2, 2, figsize=(12, 12))
#
# axs[0, 0].imshow(denoised_image1, cmap='gray')
# axs[0, 0].set_title('Image 1')
# axs[0, 0].axis('off')
#
# axs[0, 1].imshow(denoised_image2, cmap='gray')
# axs[0, 1].set_title('Image 2')
# axs[0, 1].axis('off')
#
# axs[1, 0].imshow(denoised_image3, cmap='gray')
# axs[1, 0].set_title('Image 3')
# axs[1, 0].axis('off')
#
# axs[1, 1].imshow(denoised_image4, cmap='gray')
# axs[1, 1].set_title('Image 4')
# axs[1, 1].axis('off')
#
# plt.tight_layout()
# plt.show()
# # # 显示或保存结果
# io.imshow(denoised_image1)
# io.imshow(denoised_image2)
# io.imshow(denoised_image3)
# io.imshow(denoised_image4)
# io.show()
