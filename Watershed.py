import cv2
import numpy as np
from PIL import Image


def watershed_edge(img_gray):
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    out = Image.fromarray(blur_gray)
    out.save(f"img/gray.tif")
    low_threshold = 80
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    out = Image.fromarray(edges)
    out.save(f"img/edge.tif")


def watershed(target_img):
    img_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    morph_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    background = cv2.dilate(morph_open, kernel, iterations=3)
    distance = cv2.distanceTransform(morph_open, cv2.DIST_L2, 5)
    ret, foreground = cv2.threshold(distance, 0.7 * distance.max(), 255, 0)
    foreground = np.uint8(foreground)
    unknown = cv2.subtract(background, foreground)
    ret, markers = cv2.connectedComponents(foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    watershed_marker = cv2.watershed(target_img, markers)
    return watershed_marker


if __name__ == "__main__":
    img = cv2.imread('img/lena.tif', cv2.IMREAD_COLOR)
    markers_array = watershed(img)
    img[markers_array == -1] = [0, 255, 0]
    out = Image.fromarray(img)
    out.save(f"img/watershed.tif")
