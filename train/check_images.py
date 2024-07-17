import cv2
from PIL import Image
import numpy as np
import os
import shutil

def check_and_copy_images(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_path = os.path.normpath(file_path)
            if os.path.exists(file_path):
                print("文件存在")
            else:
                print("文件不存在")
            try:
                # image = cv2.imread(file_path)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # cv2.imshow('Gray Image', image)
                # # 显示
                # cv2.namedWindow('findCorners', 0)
                # cv2.resizeWindow('findCorners', 700, 900)  # 自己设定窗口图片的大小
                # cv2.imshow("findCorners", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # print("image:", image)
                if image is not None:
                    print("succeed")
                    Image.fromarray(image)
                    shutil.copy(file_path, destination_folder)
                else:
                    print('Failed')
            except Exception as e:
                print(f"无法打开或处理 {file_path} ，错误信息: {e}")

if __name__ == '__main__':
    folder_path = r"E:\files\basic information\Q-091-20240215-6"
    dest_path = r"E:\Projects\ItS2CLR-local\base_dataset"
    check_and_copy_images(folder_path, dest_path)