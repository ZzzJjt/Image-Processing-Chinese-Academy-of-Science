import numpy as np
import cv2
import test
def isPower(n):
    if n<1:
        return False
    i = 1
    while i<=n:
        if i==n:
            return True
        i <<= 1
    return False
img = np.zeros([512,512])
img[512//2-30:512//2+30,512//2-5:512//2+5]=1
cv2.imshow('img', img)
F1 = test.dft2D(img)
# F1 = cv2.normalize(img,F1,0,1,cv2.NORM_MINMAX)
F1 = test.normalization(np.abs(F1))
img_c = np.zeros(img.shape)
cv2.imshow('F1',  F1)
for i in range (img.shape[0]):
    for j in range (img.shape[1]):
        img_c[i,j] = img[i,j]*((-1)**(i+j))
F2 = test.dft2D(img_c)
cv2.imshow('F2', test.normalization(np.abs(F2)))
F3 = np.log(1+np.abs(F2))
cv2.imshow('F3',  test.normalization(F3))
cv2.waitKey(0)
cv2.destroyAllWindows()
