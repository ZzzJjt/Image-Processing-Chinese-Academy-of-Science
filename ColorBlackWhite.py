if __name__ == '__Color to Black and White__':
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib


    # 将’NTSC’作为缺省方式
    def rgb1gray(f, method='NTSC'):

        # 防止三通道相加溢出，所以先转换类型
        f = f.astype(np.float32) / 255

        # 获取图像的三通道
        # 法一用cv：b,g,r = cv2.split(f)
        # 法二自己分离：
        b = f[:, :, 0]
        g = f[:, :, 1]
        r = f[:, :, 2]
        if method == 'average':
            gray = (r + g + b) / 3
        elif method == 'NTSC':
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # 再把类型转换回来
        gray = (gray * 255).astype('uint8')

        return gray


    if __name__ == '__Color to Black and White__':
        # 分别读取两幅图像
        img1 = cv2.imread('/Users/zhengjiatong/Desktop/mandril_color.tif', cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread('/Users/zhengjiatong/Desktop/lena512color.tiff', cv2.IMREAD_UNCHANGED)

        gray1_1 = rgb1gray(img1, 'average')
        gray1_2 = rgb1gray(img1, 'NTSC')

        gray2_1 = rgb1gray(img2, 'average')
        gray2_2 = rgb1gray(img2, 'NTSC')

        plt.figure()
        plt.subplot(1, 2, 1)
        ax = plt.gca()
        plt.axis('off')  # 去掉坐标系
        plt.imshow(gray1_1, cmap='gray')  # 显示灰度图像
        ax.set_title("average")
        plt.subplot(1, 2, 2)
        ax = plt.gca()
        plt.axis('off')  # 去掉坐标系
        plt.imshow(gray1_2, cmap='gray')
        ax.set_title("NTSC")
        plt.show()

        plt.figure()
        plt.subplot(1, 2, 1)
        ax = plt.gca()
        plt.axis('off')  # 去掉坐标系
        plt.imshow(gray2_1, cmap='gray')
        ax.set_title("average")
        plt.subplot(1, 2, 2)
        ax = plt.gca()
        plt.axis('off')  # 去掉坐标系
        plt.imshow(gray2_2, cmap='gray')
        ax.set_title("NTSC")
        plt.show()