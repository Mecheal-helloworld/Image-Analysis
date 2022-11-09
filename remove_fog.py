import cv2 as cv
import numpy as np


# 计算每个通道中的最小值，输入Image图像，输出最小值img_min
def darkChannel(img_input):
    rows, cols, channels = img_input.shape
    rtn_img = np.array(img_input)
    img_channel = img_input.min(2)
    for c in range(channels):
        rtn_img[:, :, c] = img_channel
    return rtn_img


# 最小值滤波，输入最小值图像，在2*r+1的矩形窗口内寻找最小值
def min_filter(img_input, r=2):
    rows, cols, channels = img_input.shape  # 输出为暗通道图像
    rtn_img = np.array(img_input)
    for i in range(0, rows):
        for j in range(0, cols):
            left = max(0, i - r)
            right = min(rows - 1, i + r)
            up = max(0, j - r)
            down = min(cols - 1, j + r)
            # 寻找像素点(i,j)为中心的5*5窗口内的每个通道的最小值
            rtn_img[i, j, :] = img_input[left: right, up: down, :].min(0).min(0)
    return rtn_img


# 基于导向滤波进行暗通道图像的变换
def guided_filter(img_input, p, r, eps):
    # img_input归一化之后的原图，p最小值图像，r导向滤波搜索范围，eps为惩罚项，输出导向滤波后的图像
    # q = a * I + b
    mean_I = cv.blur(img_input, (r, r))  # I的均值平滑
    mean_p = cv.blur(p, (r, r))  # p的均值平滑
    mean_II = cv.blur(img_input * img_input, (r, r))  # I*I的均值平滑
    mean_Ip = cv.blur(img_input * p, (r, r))  # I*p的均值平滑
    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p  # 协方差
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv.blur(a, (r, r))  # 对a、b进行均值平滑
    mean_b = cv.blur(b, (r, r))
    q = mean_a * img_input + mean_b
    return q


# 计算大气光a和折射图t
def select_bright(img_input, img_origin, w, t0, input_v):
    # 输入：img_input最小值图像，img_origion原图，w是t之前的修正参数，t0阈值，V导向滤波结果
    rows, cols, channels = img_input.shape
    size = rows * cols
    order = img_input[:, :, 0].reshape(size).tolist()
    order.sort(reverse=True)
    index = int(size * 0.001)  # 从暗通道中选取亮度最大的前0.1%
    mid = order[index]
    img_hsv = cv.cvtColor(img_origin, cv.COLOR_RGB2HLS)
    pos = np.argwhere(img_input[:, :, 0] >= mid)
    # pos = pos.transpose()
    # rtn_a = img_hsv[pos[0], pos[1], 1].max()
    rtn_a = 0
    for x in pos:
        rtn_a = max(rtn_a, img_hsv[x[0], x[1], 1])
    rtn_v = input_v * w
    rtn_t = 1 - rtn_v / rtn_a
    rtn_t = np.maximum(rtn_t, t0)
    return rtn_t, rtn_a


def repair(img_input, input_t, input_a):
    input_t = input_t - 0.25  # 不知道为什么这里减掉0.25效果才比较好
    rtn_j = (img_input - (input_a / 255.0)) / input_t + (input_a / 255.0)
    return rtn_j


if __name__ == "__main__":
    img = cv.imread('C:\\Users\\Administrator\\Desktop\\R-C-1.jpg')
    img_arr = np.array(img / 255.0)  # 归一化
    img_min = darkChannel(img_arr)  # 计算每个通道的最小值
    img_dark = min_filter(img_min, 2)  # 计算暗通道图像
    img_guided = guided_filter(img_arr, img_min, r=75, eps=0.001)
    t, A = select_bright(img_min, img, w=0.95, t0=0.1, input_v=img_guided)
    img_repair = repair(img_arr, t, A)
    cv.imshow('Origin', img)
    cv.imshow('DarkChannel', img_dark)
    cv.imshow('Repair', img_repair)
    cv.waitKey()
    cv.destroyAllWindows()
