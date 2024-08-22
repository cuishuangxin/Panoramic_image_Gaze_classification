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


def plot_scanpath(img,fx, fy, sx,sy, bbox=None, title=None,img_name=None,number=0 ):

    # print(img.shape)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ########
    # # print(H, W)
    # create_green = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # create_green[:, :, 0] = 0
    # create_green[:, :, 1] = 0  # 这里我创建一个纯绿色的图像
    # create_green[:, :, 2] = 255
    # print(image.shape), create_green.shape
    # image = cv2.addWeighted(image, 0.7, create_green, 0.3, 0)

    b, g, r = cv2.split(img)
    image_rgb = cv2.merge((r, g, b))
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)




    for i in range(len(sx)):
        # cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        cir_rad = 40
        circle = plt.Circle((sx[i], sy[i]),
                            radius=cir_rad,
                            facecolor="#F4B71D",
                            alpha=0.35)
        ax.add_patch(circle)

    # for i in range(len(fx)):
    #     if i > 0:
    #         plt.arrow(fx[i - 1], fy[i - 1], fx[i] - fx[i - 1],
    #                   fy[i] - fy[i - 1], width=3, color='red', alpha=0.6)

    for i in range(len(fx)):
        # cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        cir_rad = 60
        circle = plt.Circle((fx[i], fy[i]),
                            radius=cir_rad,
                            facecolor='#FF0000',
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
    plt.savefig('D:\csx\ScanPath\AOI\\newgraph\\f'+img_name+'_%02d' %number+'.jpg',bbox_inches='tight',dpi = 500,pad_inches=0.0)
    plt.close('all')
    # plt.show()



def splot_scanpath(img,fx, fy, sx,sy, bbox=None, title=None,img_name=None,number=0 ):
    b, g, r = cv2.split(img)
    image_rgb = cv2.merge((r, g, b))
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    for i in range(len(fx)):
        # cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        cir_rad = 60
        circle = plt.Circle((fx[i], fy[i]),
                            radius=cir_rad,
                            facecolor='#FF0000',
                            alpha=0.1)       #######  '#F49215'
        ax.add_patch(circle)
    for i in range(len(sx)):
        # cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        cir_rad = 40
        circle = plt.Circle((sx[i], sy[i]),
                            radius=cir_rad,
                            facecolor="#F4B71D",
                            alpha=1)       #######  "#F4B71D"
        ax.add_patch(circle)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.savefig('D:\csx\ScanPath\AOI\\newgraph\\s'+img_name+'_%02d' %number+'.jpg',bbox_inches='tight',dpi = 500,pad_inches=0.0)
    plt.close('all')
    # plt.show()

def mplot_scanpath(img, fx, fy, sx, sy, bbox=None, title=None, img_name=None, number=0):
    b, g, r = cv2.split(img)
    image_rgb = cv2.merge((r, g, b))
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    for i in range(len(sx)):
        # cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        cir_rad = 60
        circle = plt.Circle((sx[i], sy[i]),
                            radius=cir_rad,
                            facecolor="#F4B71D",
                            alpha=0.8)  #######  "#F4B71D"
        ax.add_patch(circle)
    for i in range(len(fx)):
        # cir_rad = int(25 + rad_per_T * (ts[i] - min_T))
        cir_rad = 60
        circle = plt.Circle((fx[i], fy[i]),
                            radius=cir_rad,
                            facecolor='#FF0000',
                            alpha=0.8)  #######  '#F49215'
        ax.add_patch(circle)

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.savefig('D:\csx\ScanPath\AOI_600v2\output\output\\m' + img_name + '_%02d' % number + '.jpg', bbox_inches='tight',
                dpi=500, pad_inches=0.0)
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
    path1 = r'D:\csx\ScanPath\AOI_600v2\1_150'
    path2 = r'D:\csx\ScanPath\AOI_600v2\2_50'
    path3 = r'D:\csx\ScanPath\AOI_600v2\1.5_100'
    for image_file in os.listdir('D:\csx\ScanPath\AOI_600v2\ceshi'):
        img_path = os.path.join('D:\csx\ScanPath\AOI_600v2\ceshi', image_file)
        img = cv2.imread(img_path)

        path_list = os.listdir(os.path.join(path1, image_file[:-4]))
        path_list.sort(key=lambda x: int(x[:-4]))
        n = 1
        for t_name in path_list:
            # print(t_name)
            # 中间数据
            data = []
            # with open(os.path.join(path1, image_file[:-4], t_name), 'r') as f:
            #     reader = f.readlines()
            #     reader = reader[29:-1]  # 前五个数据没有动去掉，中间数据会多一行特征（速度）
            #     FX = []
            #     FY = []
            #     SX = []
            #     SY = []
            #     for readline in reader:
            #         # print(readline)
            #         content = readline.replace('\n', '').split(',')
            #         if content[7] == 'fixation':
            #             FX.append(int(float(content[1])))
            #             FY.append(int(float(content[2])))
            #         else:
            #             SX.append(int(float(content[1])))
            #             SY.append(int(float(content[2])))
            # plot_scanpath(img, FX, FY, SX, SY, None, None, image_file[:-4], n)
            # n = n + 1

            # with open(os.path.join(path2, image_file[:-4], t_name), 'r') as s:
            #     reader = s.readlines()
            #     reader = reader[29:-1]
            #     FX = []
            #     FY = []
            #     SX = []
            #     SY = []
            #     for readline in reader:
            #         content = readline.replace('\n','').split(',')
            #         if content[7] == "unassigned":
            #             SX.append(int(float(content[1])))
            #             SY.append(int(float(content[2])))
            #         else:
            #             FX.append(int(float(content[1])))
            #             FY.append(int(float(content[2])))
            # splot_scanpath(img,FX[::5],FY[::5],SX,SY,None,None,image_file[:-4],n)
            # n = n + 1


#################生成中间Medium的轨迹图
            FX = []
            FY = []
            SX = []
            SY = []
            UX = []
            UY = []
            with open(os.path.join(path3, image_file[:-4], t_name), 'r') as f:
                reader = f.readlines()
                reader = reader[29:-1]  # 前五个数据没有动去掉，中间数据会多一行特征（速度）
                index1 = []
                num = 0
                for readline in reader:
                    # print(readline)
                    num += 1
                    if num % 3 == 0:
                        content = readline.replace('\n', '').split(',')
                        index1.append(content[7])
                        data.append((content[0] + ';' + content[1] + ';' + content[2] + ';' + content[8]))

            with open(os.path.join(path1, image_file[:-4], t_name), 'r') as f:
                reader = f.readlines()
                reader = reader[28:-1]
                index2 = []
                for readline in reader:
                    content = readline.replace('\n', '').split(',')
                    index2.append(content[7])

            with open(os.path.join(path2, image_file[:-4], t_name), 'r') as f:
                reader = f.readlines()
                reader = reader[28:-1]
                index3 = []
                for readline in reader:
                    content = readline.replace('\n', '').split(',')
                    index3.append(content[7])

            for id in range(len(data)):
                value = data[id].split(';')
                coord_x = float(value[1])
                coord_y = float(value[2])

                # print(index1[id],index2[id],index3[id])

                # if index1[id] == 'fixation' and index1[id] == index2[id]:
                #     FX.append(coord_x)
                #     FY.append(coord_y)
                # elif index1[id] == 'unassigned' and index1[id] == index3[id]:
                #     SX.append(coord_x)
                #     SY.append(coord_y)
                # elif (index1[id] == 'fixation' and index2[id] == 'unassigned') or (
                #         index1[id] == 'unassigned' and index3[id] == 'fixation'):
                #     UX.append(coord_x)
                #     UY.append(coord_y)

                if index1[id] == 'fixation':
                    FX.append(coord_x)
                    FY.append(coord_y)
                else:
                    SX.append(coord_x)
                    SY.append(coord_y)


            mplot_scanpath(img,FX[::5],FY[::5],SX,SY,None,None,image_file[:-4],n)
            n = n + 1





