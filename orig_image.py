#coding:utf-8
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL.Image import Image
from matplotlib.patches import Rectangle
import numpy as np
import argparse
import json
from os.path import isfile


def convert_coordinate(X, Y, im_w, im_h):
    """
    convert from display coordinate to pixel coordinate
    X - x coordinate of the fixations
    Y - y coordinate of the fixations
    im_w - image width
    im_h - image height
    """
    display_w, display_h = 1680, 1050
    target_ratio = display_w / float(display_h)
    ratio = im_w / float(im_h)

    delta_w, delta_h = 0, 0
    if ratio > target_ratio:
        new_w = display_w
        new_h = int(new_w / ratio)
        delta_h = display_h - new_h
    else:
        new_h = display_h
        new_w = int(new_h * ratio)
        delta_w = display_w - new_w
    dif_ux = delta_w // 2
    dif_uy = delta_h // 2
    scale = im_w / float(new_w)
    X = (X - dif_ux) * scale
    Y = (Y - dif_uy) * scale
    print("X",X)
    print("Y",Y)
    return X, Y


def plot_scanpath(img,xs, ys, bbox=None, title=None,img_name=None,number=0 ):
    b, g, r = cv2.split(img)
    image_rgb = cv2.merge((r, g, b))
    fig, ax = plt.subplots()

    ax.imshow(image_rgb)
    cir_rad_min, cir_rad_max = 30, 60



    for i in range(len(xs)):
        if abs(xs[i] - xs[i - 1])>500:
            continue
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1],
                      ys[i] - ys[i - 1], width=3, color='red', alpha=0.6)

    for i in range(len(xs)):
        # cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        cir_rad = 50
        circle = plt.Circle((xs[i], ys[i]),
                            radius=cir_rad,
                            facecolor='yellow',
                            alpha=0.8)
        ax.add_patch(circle)
        # plt.annotate("{}".format(
        #     i + 1), xy=(xs[i], ys[i] + 3), fontsize=4, ha="center", va="center")


    # if bbox is not None:
    #     rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
    #                      alpha=0.5, edgecolor='yellow', facecolor='none', linewidth=2)
    #     ax.add_patch(rect)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.savefig('D:\csx\ScanPath\AOI_600v2\output\\1all_'+img_name+'_%02d' %number+'.jpg',pad_inches=0, bbox_inches='tight',dpi = 500)
    plt.close('all')
    # plt.show()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixation_path', type=str, help='the path of the fixation json file')
    parser.add_argument('--image_dir', type=str, help='the directory of the image stimuli')
    parser.add_argument('--random_trial', choices=[0, 1],
                        default=1, type=int, help='randomly drawn from data (default=1)')
    parser.add_argument('--trial_id', default=0, type=int, help='trial id (default=0)')
    parser.add_argument('--subj_id', type=int, default=-1,
                        help='subject id (default=-1)')
    parser.add_argument('--task',
                        choices=['bottle', 'chair', 'cup', 'fork', 'bowl', 'mouse',
                                 'microwave', 'laptop', 'key', 'sink', 'toilet', 'clock', 'tv',
                                 'stop sign', 'car', 'oven', 'knife'],
                        default='bottle',
                        help='searching target')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # load image
    path = r'D:\csx\ScanPath\AOI_600v2\1.5_100'
    for image_file in os.listdir('D:\csx\ScanPath\AOI_600v2\ceshi'):
        image_file = image_file[:-4]
        img_path = os.path.join('D:\csx\ScanPath\AOI_600v2\ceshi', image_file+'.jpg')
        print(img_path)

        # img_height, img_width = 2048, 4096
        # img = cv2.resize(cv2.imread(img_path), (img_width, img_height))
        img = cv2.imread(img_path)
        n = 1
        path_list = os.listdir(os.path.join(path, image_file))
        path_list.sort(key=lambda x: int(x[:-4]))

        for data_file in path_list:
            # if n > 15:
            #     continue
            print(data_file)
            scanFile = open(os.path.join(path, image_file, data_file), 'r')  # 打开文件

            file_data = scanFile.readlines()  # 读取所有行
            file_data = file_data[29:-1]
            X = []
            Y = []
            num = 0
            for row in file_data:
                tmp_list = row.replace('\n','').split(',')
                num += 1
                if num%3 == 0:
                # if tmp_list[7] == 'fixation':
                    X.append(int(float(tmp_list[1])))
                    Y.append(int(float(tmp_list[2])))
            plot_scanpath(img, X, Y, None, None, image_file, n)
            n = n + 1
            print(len(X))

# load image
#     path = r'D:\csx\ScanPath\AOI_600v2\ablation_study\ED_SLIC_TOPn\PoSample\data'
#     for image_file in os.listdir('D:\csx\ScanPath\AOI_600v2\ceshi1'):
#         img_path = os.path.join('D:\csx\ScanPath\AOI_600v2\ceshi1', image_file)
#         img = cv2.imread(img_path)
#         # img_height, img_width = 2048, 4096
#         # img = cv2.resize(cv2.imread(img_path), (img_width, img_height))
#         n = 1
#         scanFile = open(os.path.join(path, image_file[:-4]+'.csv'), 'r')  # 打开文件
#
#         file_data = scanFile.readlines()  # 读取所有行
#         for row in file_data:
#             X = []
#             Y = []
#             row = row.replace(" ", "").replace("\n", "").split(',')
#             for i in range(0,len(row)):
#                 tmp = row[i].split(';')
#                 X.append(int(float(tmp[1])))
#                 Y.append(int(float(tmp[2])))
#             plot_scanpath(img, X, Y, None, None, image_file[:-4], n)
#             n = n + 1




