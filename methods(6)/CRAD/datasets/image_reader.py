import os
from itertools import count

import cv2
import numpy as np


class OpenCVReader:
    def __init__(self, image_dir, color_mode):
        self.image_dir = image_dir
        self.color_mode = color_mode
        assert color_mode in ["RGB", "BGR", "GRAY"], f"{color_mode} not supported"
        if color_mode != "BGR":
            self.cvt_color = getattr(cv2, f"COLOR_BGR2{color_mode}")
        else:
            self.cvt_color = None

    def __call__(self, filename, is_mask=False):
        filename = os.path.join(self.image_dir, filename)
        assert os.path.exists(filename), filename
        # # 检查文件是否存在
        # if not os.path.exists(filename):
        #     print(f"Warning: {filename} does not exist. Skipping.")
        #     # return np.zeros((100, 100, 3), dtype=np.uint8)  # 返回一个占位符图像
        # else:
        if is_mask:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        # 检查读取结果
        if img is None:
            print(f"Error: Failed to load image from------------------- {filename}. -----------------------------------Returning placeholder.")
            return np.zeros((100, 100, 3), dtype=np.uint8)  # 返回占位符图像

        # 打印图像的维度，方便调试
        print(f"Image loaded successfully: {img.shape}")
        if self.color_mode != "BGR":
            try:
                img = cv2.cvtColor(img, self.cvt_color)
            except cv2.error as e:
                print(f"Color conversion failed for------------------------------------- {filename}: {e}")
                return np.zeros((100, 100, 3), dtype=np.uint8)  # 返回占位符图像
        return img



def build_image_reader(cfg_reader):
    if cfg_reader["type"] == "opencv":
        return OpenCVReader(**cfg_reader["kwargs"])
    else:
        raise TypeError("no supported image reader type: {}".format(cfg_reader["type"]))
