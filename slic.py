import csv
import os

import cv2
import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import directed_hausdorff, euclidean

import suppor_lib


def compare_zeros_ones(array):
    count_zeros = array.count(0)
    count_ones = array.count(1)

    if count_zeros > count_ones:
        return 0
    elif count_ones > count_zeros:
        return 1
    else:
        # 如果0和1的个数相等，可以根据实际需求返回任何值，这里返回None
        return None
def DTW(P, Q, **kwargs):
    dist, _ =  fastdtw(P, Q, dist=euclidean)
    return dist


# 计算两点之间的欧氏距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# 找到短轨迹到长轨迹的最短距离
def find_shortest_distance(short_track, long_track):
    window_size = 10
    min_distance = float('inf')

    for i in range(0,len(long_track),10):
        window = long_track[i:i + window_size]

        # 计算滑动窗口内短轨迹点到滑动窗口内长轨迹线段的距离之和
        distance = DTW(short_track, window)

        if distance < min_distance:
            min_distance = distance

    return min_distance
#第一次
# orig_index_file = r'D:\csx\ScanPath\AOI_300\data\Uncertain\index'
# new_index_file = r'D:\csx\ScanPath\AOI_300\data\Uncertain\SLIC\index'
# orig_data_file = r'D:\csx\ScanPath\AOI_300\data\Uncertain\data'
# orig_label_file = r'D:\csx\ScanPath\AOI_300\data\Uncertain\label'
# cluster_file = r'D:\csx\ScanPath\AOI_300\data\Uncertain\SLIC\cluster'
#
# positive_path = r'D:\csx\ScanPath\ScanDMM-master\demo\AOI_300\490+'
# negative_path = r'D:\csx\ScanPath\ScanDMM-master\demo\AOI_300\260-'
# image_path = r'D:\csx\ScanPath\ScanDMM-master\demo\input'
# save_path = r'D:\csx\ScanPath\AOI_300\data\epoch1'

#第二次
orig_index_file = r'D:\csx\ScanPath\AOI_300\data\epoch1\Uncertain\index'
new_index_file = r'D:\csx\ScanPath\AOI_300\data\epoch1\Uncertain\SLIC\index'
orig_data_file = r'D:\csx\ScanPath\AOI_300\data\epoch1\Uncertain\data'
orig_label_file = r'D:\csx\ScanPath\AOI_300\data\epoch1\Uncertain\label'
cluster_file = r'D:\csx\ScanPath\AOI_300\data\SLIC1\cluster'

positive_path = r'D:\csx\ScanPath\ScanDMM-master\demo\AOI_300\2_400+'
negative_path = r'D:\csx\ScanPath\ScanDMM-master\demo\AOI_300\2_280-'
image_path = r'D:\csx\ScanPath\ScanDMM-master\demo\input'
save_path = r'D:\csx\ScanPath\AOI_300\data\epoch2'




if not os.path.exists(os.path.join(save_path, "PoSample", "data")):
    os.makedirs(os.path.join(save_path, "PoSample", "data"))
if not os.path.exists(os.path.join(save_path, "PoSample", "index")):
    os.makedirs(os.path.join(save_path, "PoSample", "index"))
if not os.path.exists(os.path.join(save_path, "NeSample", "data")):
    os.makedirs(os.path.join(save_path, "NeSample", "data"))
if not os.path.exists(os.path.join(save_path, "NeSample", "index")):
    os.makedirs(os.path.join(save_path, "NeSample", "index"))
if not os.path.exists(os.path.join(save_path, "Uncertain", "data")):
    os.makedirs(os.path.join(save_path, "Uncertain", "data"))
if not os.path.exists(os.path.join(save_path, "Uncertain", "index")):
    os.makedirs(os.path.join(save_path, "Uncertain", "index"))
if not os.path.exists(os.path.join(save_path, "Uncertain", "label")):
    os.makedirs(os.path.join(save_path, "Uncertain", "label"))



for image_name in os.listdir(cluster_file):
    positive = np.load(os.path.join(positive_path, image_name[:-4] + '.npy'))  # 预测的正轨迹
    negative = np.load(os.path.join(negative_path, image_name[:-4] + '.npy'))  # 预测的负轨迹
    image = cv2.imread(os.path.join(image_path, image_name[:-4] + '.jpg'))


    num_interpolated_points = 20  # 每两点之间插值的点数
    Po_scanpaths = []
    for j in range(positive.shape[0]):
        interpolated_points = []
        for i in range(positive.shape[1] - 1):
            for t in np.linspace(0, 1, num_interpolated_points):
                interpolated_point = (1 - t) * positive[j][i] + t * positive[j][i + 1]
                interpolated_points.append(interpolated_point)
        Po_scanpaths.append(torch.tensor(np.array(interpolated_points)))
    Po_scanpaths = torch.stack(Po_scanpaths)

    Ne_scanpaths = []
    for j in range(negative.shape[0]):
        interpolated_points = []
        for i in range(negative.shape[1] - 1):
            for t in np.linspace(0, 1, num_interpolated_points):
                interpolated_point = (1 - t) * negative[j][i] + t * negative[j][i + 1]
                interpolated_points.append(interpolated_point)
        # tt = suppor_lib.xyz2plane(torch.tensor(np.array(interpolated_points)))
        Ne_scanpaths.append(torch.Tensor(np.array(interpolated_points)))
    Ne_scanpaths = torch.stack(Ne_scanpaths)




    f =  open(os.path.join(cluster_file,image_name),'r')  #聚类中心
    f = f.readlines()
    p = open(os.path.join(new_index_file,image_name),'r') #新数据index
    p = p.readlines()

    g = open(os.path.join(orig_data_file,image_name),'r') # 原数据data
    g = g.readlines()
    w = open(os.path.join(orig_index_file,image_name),'r')# 原数据index
    w = w.readlines()
    o = open(os.path.join(orig_label_file,image_name),'r') # 源数据label
    o = o.readlines()

    for cluster,new_index,orig_data,orig_index,orig_label in zip(f,p,g,w,o):
        cluster = cluster.replace('\n','').split(',')
        new_index = new_index.replace('\n','').split(',')
        orig_data = orig_data.replace('\n','').split(',')
        orig_index = orig_index.replace('\n','').split(',')
        orig_label = orig_label.replace('\n','').split(',')
        new_fixation_data = []
        new_fixation_index = []
        new_saccade_data = []
        new_saccade_index = []
        new_uncertain_data = []
        new_uncertain_index = []
        new_uncertain_label = []
        if len(cluster) > 1:
            for id in range(len(cluster)):
                cluster_point = cluster[id].split(';')[0]
                cluster_label = cluster[id].split(';')[3]
                series = []
                series_orig = []
                index = []
                label = []
                for k in range(len(new_index)):
                    if new_index[k] == cluster_point: # 如果当前点属于该cluster
                        content = orig_data[k].split(';')
                        #要验证的轨迹片段
                        series_orig.append(orig_data[k])
                        series.append([float(content[2])/image.shape[0]*180-90,float(content[1])/image.shape[1]*360-180])  #坐标点序列
                        index.append(orig_index[k]) # 下标序列
                        label.append(orig_label[k]) # 标签
                if len(series) <= 3:
                    continue
                tmp_gaze = torch.Tensor(series)
                tmp_gaze = suppor_lib.sphere2xyz(tmp_gaze)
                dist = []
                predict_label = []
                for i in range(positive.shape[0]):
                    min_distance = find_shortest_distance(tmp_gaze, Po_scanpaths[i])
                    # dist.append(DTW(tmp_gaze,positive[i]))
                    dist.append(min_distance)
                    predict_label.append(1)

                for j in range(negative.shape[0]):
                    min_distance = find_shortest_distance(tmp_gaze,Ne_scanpaths[j])
                    # dist.append(DTW(tmp_gaze,negative[j]))
                    dist.append(min_distance)
                    predict_label.append(0)

                dist = np.array(dist)

                predict_label = np.array(predict_label)

                # 获取每个值及其索引的对应关系
                values_with_indices = [(value, index) for index, value in enumerate(dist)]
                # 按值进行排序
                values_with_indices.sort()

                # 提取前五个最小值的索引
                top_5_indices = [index for value, index in values_with_indices[:5]]
                predict_label = predict_label[top_5_indices]

                predict_label = [x for x in predict_label]
                result = compare_zeros_ones(predict_label)

                if result == int(float(cluster_label)) and result == 1:
                    for item_data,item_index in zip(series_orig,index):
                        new_fixation_data.append(item_data)
                        new_fixation_index.append(item_index)
                elif result == int(float(cluster_label)) and result == 0:
                    for item_data,item_index in zip(series_orig,index):
                        new_saccade_data.append(item_data)
                        new_saccade_index.append(item_index)
                else:
                    for item_data,item_index,item_label in zip(series_orig,index,label):
                        new_uncertain_data.append(item_data)
                        new_uncertain_index.append(item_index)
                        new_uncertain_label.append(item_label)



                # min_value = min(dist)
                # min_index = dist.index(min_value)
                # print(min(zheng), min(fu),min_index)
                #
                # if predict_label[min_index] == int(float(cluster_label)) and predict_label[min_index] == 1:
                #     for item_data,item_index in zip(series_orig,index):
                #         new_fixation_data.append(item_data)
                #         new_fixation_index.append(item_index)
                #     print("fixation")
                # elif predict_label[min_index] == int(float(cluster_label)) and predict_label[min_index] == 0:
                #     for item_data,item_index in zip(series_orig,index):
                #         new_saccade_data.append(item_data)
                #         new_saccade_index.append(item_index)
                #     print("saccade")
                # else:
                #     for item_data,item_index,item_label in zip(series_orig,index,label):
                #         new_uncertain_data.append(item_data)
                #         new_uncertain_index.append(item_index)
                #         new_uncertain_label.append(item_label)
                #     print("no")

        print(len(orig_data),len(new_fixation_data),len(new_saccade_data),len(new_uncertain_data))
        with open(os.path.join(save_path,"PoSample","data",image_name), 'a', newline='') as csvfile1:
            writer = csv.writer(csvfile1)
            writer.writerow(new_fixation_data)
        with open(os.path.join(save_path, "PoSample", "index", image_name), 'a', newline='') as csvfile2:
            writer = csv.writer(csvfile2)
            writer.writerow(new_fixation_index)

        with open(os.path.join(save_path,"NeSample","data",image_name),'a',newline='') as csvfile3:
            writer = csv.writer(csvfile3)
            writer.writerow(new_saccade_data)
        with open(os.path.join(save_path,"NeSample","index",image_name),'a',newline='') as csvfile4:
            writer = csv.writer(csvfile4)
            writer.writerow(new_saccade_index)

        with open(os.path.join(save_path,"Uncertain","data",image_name),'a',newline='') as csvfile5:
            writer = csv.writer(csvfile5)
            writer.writerow(new_uncertain_data)
        with open(os.path.join(save_path,"Uncertain","index",image_name),'a',newline='') as csvfile6:
            writer = csv.writer(csvfile6)
            writer.writerow(new_uncertain_index)
        with open(os.path.join(save_path,"Uncertain","label",image_name),'a',newline='') as csvfile7:
            writer = csv.writer(csvfile7)
            writer.writerow(new_uncertain_label)
