import numpy as np
from PIL import Image


def spatial_resolution():
    img = Image.open("img/lena.tif")
    out = img.resize((int(img.size[0] / 2), int(img.size[1] / 2)), Image.Resampling.LANCZOS)
    out.save(f"img/lena{out.size[0]}x{out.size[1]}.tif")
    out = img.resize((int(img.size[0] / 4), int(img.size[1] / 4)), Image.Resampling.LANCZOS)
    out.save(f"img/lena{out.size[0]}x{out.size[1]}.tif")
    out = img.resize((int(img.size[0] / 8), int(img.size[1] / 8)), Image.Resampling.LANCZOS)
    out.save(f"img/lena{out.size[0]}x{out.size[1]}.tif")


def gray_resolution():
    img = Image.open("img/lena.tif")
    img_array = np.array(img)
    out_array = img_array >> 2
    out = Image.fromarray(out_array)
    out.save(f"img/lena_0-127.tif")
    out_array = img_array >> 3
    out = Image.fromarray(out_array)
    out.save(f"img/lena_0-63.tif")
    out_array = img_array >> 4
    out = Image.fromarray(out_array)
    out.save(f"img/lena_0-31.tif")


if __name__ == '__main__':
    spatial_resolution()
    gray_resolution()
