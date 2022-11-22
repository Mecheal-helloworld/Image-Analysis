import cv2
import numpy as np


def OTSU(img_gray, th_begin=0, th_end=256, th_step=1):
    max_g = 0
    suitable_th = 0
    for threshold in range(th_begin, th_end, th_step):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if not fore_pix:
            break
        if not back_pix:
            continue
        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold
    return suitable_th


def iterative_segmentation(img_gray):
    threshold = (int(img_gray.max()) + int(img_gray.min())) / 2
    while True:
        img_upper = np.where(img_gray > threshold, 1, img_gray)  # 大于某个值的元素由0替代
        img_lower = np.where(img_gray < threshold, 0, img_gray)  # 小于某个值的元素由0替代
        upper = np.sum(img_upper) / np.sum(img_upper != 0)
        lower = np.sum(img_lower) / np.sum(img_lower != 0)
        threshold_next = (upper + lower) / 2
        if abs(threshold - threshold_next) < 0.01:
            return threshold
        else:
            threshold = threshold_next


if __name__ == "__main__":
    img = cv2.imread('img/lena.tif', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_array = np.array(gray)
    th = OTSU(gray_array)
    # 根据最新的阈值进行分割
    gray_array[gray_array > th] = 255
    gray_array[gray_array <= th] = 0
