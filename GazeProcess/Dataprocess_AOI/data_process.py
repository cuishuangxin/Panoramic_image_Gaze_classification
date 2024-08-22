import cv2
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import os
import config
import suppor_lib
import pickle
import pickle as pck
torch.set_printoptions(threshold=np.inf)
'''
    Description: data_process.py is used to obtain Ground-truth scanpaths 
    for model training and validation. We give an example of how we process 
    Sitzmann dataset (https://github.com/vsitzmann/vr-saliency). 

    We provide ready-to-use training & validation data in './Datasets/Sitzmann.pkl'
    You can also follow the 3 steps to reproduce the dataset:

        # 1. Rotate 360-degree images for data augmentation using:
             suppor_lib.rotate_images(input_path, output_path)

        # 2. Modify the configure in config.py

        # 3. Modify and run data_process.py.

    Data format:

    [data]
        ['train']
            ['image1_name']
                ['image']: Tensor[3, 128, 256]
                ['scanpaths']: Tensor[n_scanpath, n_gaze_point, 3] # (x, y, z) for the 3-th dimension
            ['image2_name']
                ...
        ['test']
            ['imageN_name']
                ...
        ['info']
            ['train']: {
                'num_image': int,
                'num_scanpath': int,
                'scanpath_length': int,
                'max_scan_length': int
            }
            ['test']: {
                ...
            }
'''


def save_file(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_logfile(path):
    log = pck.load(open(path, 'rb'), encoding='latin1')
    return log


def twoDict(pack, key_a, key_b, data):
    if key_a in pack:
        pack[key_a].update({key_b: data})
    else:
        pack.update({key_a: {key_b: data}})
    return pack


def create_info():
    info = {
        'train': {
            'num_image': 0,
            'num_scanpath': 0,
            'scanpath_length': 0,
            'max_scan_length': 0
        },
        'test': {
            'num_image': 0,
            'num_scanpath': 0,
            'scanpath_length': 0,
            'max_scan_length': 0
        }
    }
    return info


def summary(info):
    print("\n============================================")

    print("Train_set:   {} images, {} scanpaths,  length ={}, max_length ={}".
          format(info['train']['num_image'], info['train']['num_scanpath'], info['train']['scanpath_length'],info['train']['max_scan_length']))

    print("Test_set:    {} images, {} scanpaths,  length ={},max_length ={}".
          format(info['test']['num_image'], info['test']['num_scanpath'], info['test']['scanpath_length'], info['test']['max_scan_length']))

    print("============================================\n")


def forward(database_name: str):
    if not os.path.exists('Datasets'):
        os.makedirs('Datasets')

    print('\nBegin process {} database'.format(database_name))

    if database_name == 'AOI':
        data = Sitzmann_Dataset()
        dic = data.run()

        save_file('D:\csx\ScanPath\ScanDMM-master\PM_ablation\ED+TOP13\dataset\AOI600v2_3+.pkl', dic)
        summary(dic['info'])

    else:
        print('\nYou need to prepare the code for {} data processing'.format(database_name))


class Sitzmann_Dataset():
    def __init__(self):
        super().__init__()
        self.images_path = config.dic_Sitzmann['IMG_PATH']
        self.gaze_path = config.dic_Sitzmann['GAZE_PATH']
        self.test_set = config.dic_Sitzmann['TEST_SET']
        self.duration = 35
        self.info = create_info()
        self.images_test_list = []
        self.images_train_list = []
        self.image_and_scanpath_dict = {}

    def mod(self, a, b):
        c = a // b
        r = a - c * b
        return r

    def rotate(self, lat_lon, angle):
        # We convert [-180, 180] to [0, 360], then compute the new longitude.
        # We ``minus`` the angle here, which is different from what we do in rotating images,
        # because ffmepeg has a different coordination. For example, set ``yaw=60`` in ffmepg
        # equate to longitude = -60 + longitude
        new_lon = self.mod(lat_lon[:, 1] + 180 - angle, 360) - 180
        rotate_lat_lon = lat_lon
        rotate_lat_lon[:, 1] = new_lon
        return rotate_lat_lon

    def handle_empty(self, sphere_coords):
        empty_index = np.where(sphere_coords[:, 0] == -999)[0]
        throw = False
        for _index in range(empty_index.shape[0]):
            # if not throw the scanpath of this user
            if not throw:
                # if the first one second is empty
                if empty_index[_index] == 0:
                    # if the next second is not empty
                    if sphere_coords[empty_index[_index] + 1, 0] != -999:
                        sphere_coords[empty_index[_index], 0] = sphere_coords[empty_index[_index] + 1, 0]
                        sphere_coords[empty_index[_index], 1] = sphere_coords[empty_index[_index] + 1, 1]
                    else:
                        throw = True
                        # print(" Too many invalid gaze points !! {}".format(empty_index))

                # if the last one second is empty
                elif empty_index[_index] == (self.duration - 1):
                    sphere_coords[empty_index[_index], 0] = sphere_coords[empty_index[_index] - 1, 0]
                    sphere_coords[empty_index[_index], 1] = sphere_coords[empty_index[_index] - 1, 1]

                else:
                    prev_x = sphere_coords[empty_index[_index] - 1, 1]
                    prev_y = sphere_coords[empty_index[_index] - 1, 0]
                    next_x = sphere_coords[empty_index[_index] + 1, 1]
                    next_y = sphere_coords[empty_index[_index] + 1, 0]

                    if prev_x == -999 or next_x == -999:
                        throw = True
                        # print(" Too many invalid gaze points !! {}".format(empty_index))

                    else:
                        " Interpolate on lat "
                        sphere_coords[empty_index[_index], 0] = 0.5 * (prev_y + next_y)

                        " Interpolate on lon "
                        # the maximum distance between two points on a sphere is pi
                        if np.abs(next_x - prev_x) <= 180:
                            sphere_coords[empty_index[_index], 1] = 0.5 * (prev_x + next_x)
                        # jump to another side
                        else:
                            true_distance = 360 - np.abs(next_x - prev_x)
                            if next_x > prev_x:
                                _temp = prev_x - true_distance / 2
                                if _temp < -180:
                                    _temp = 360 + _temp
                            else:
                                _temp = prev_x + true_distance / 2
                                if _temp > 180:
                                    _temp = _temp - 360
                            sphere_coords[empty_index[_index], 1] = _temp

        return sphere_coords, throw



    def sample_gaze_points(self, raw_data):
        fixation_coords = []
        samples_per_bin = raw_data.shape[0] // self.duration
        bins = raw_data[:samples_per_bin * self.duration].reshape([self.duration, -1, 2])
        for bin in range(self.duration):
            " filter out invalid gaze points "
            _fixation_coords = bins[bin, np.where((bins[bin, :, 0] != 0) & (bins[bin, :, 1] != 0))]
            if _fixation_coords.shape[1] == 0:
                " mark the empty set"
                fixation_coords.append([-999, -999])
            else:
                " sample the first element in a set of one-second gaze points "
                sample_vale = _fixation_coords[0, 0]
                fixation_coords.append(sample_vale)
        sphere_coords = np.vstack(fixation_coords) - [90, 180]

        return sphere_coords


    def get_train_set(self):
        #第一次训练
        gaze_path = r'D:\csx\ScanPath\AOI_600v2\Ablation\PM_ablation\ED+TOP13\merge2\PoSample\traindata'
        image_path = r'D:\csx\ScanPath\AOI_600v2\image_train'

        max = 30
        scan_length = []
        image_id = 0
        scanpath_num = []
        for file_name in os.listdir(gaze_path):
            n = 0
            p = open(os.path.join(gaze_path, file_name), 'r')
            for gaze_row in p:
                # gaze_row = gaze_row.replace(" ", "").replace("\n", "").split(",")
                n += 1
            scanpath_num.append(n)
        print("scanpath_num",scanpath_num)

        t = 0
        for file_name in os.listdir(gaze_path):
            scanpath_id = 0
            temple_gaze = np.zeros((scanpath_num[t], max, 2)) #每一张图片的所有user
            t += 1
            f = open(os.path.join(gaze_path,file_name))
            scanpath_length = []
            # print(os.path.join(image_path, file_name[:-4]+'.jpg'))
            img = cv2.imread(os.path.join(image_path, file_name[:-4]+'.jpg'), cv2.IMREAD_COLOR)
            image = suppor_lib.image_process(os.path.join(image_path, file_name[:-4]+'.jpg'))
            for gazes_row in f:                 #每一张图片的每一个user
                gaze_row = gazes_row.replace(" ", "").replace("\n", "").replace("\"", "").split(",")
                gaze = np.full((max,2), 0)
                scanpath_length.append(int(len(gaze_row)/2))
                for j in range(0,int(len(gaze_row)/2)):
                    # lat = float(gaze_row[j*2+1])/2048*180 - 90 #纬度
                    # lon = float(gaze_row[j*2])/4096*360 - 180 #经度
                    # print(gaze_row[j*2+1])
                    lat = float(gaze_row[j*2+1])/img.shape[0]*180 - 90     #y[-90,90]
                    lon = float(gaze_row[j*2])/img.shape[1]*360 - 180   #x[-180,180]
                    gaze[j] = [lat,lon]
                    # print(gaze[j])
                temple_gaze[scanpath_id] = torch.from_numpy(gaze)
                scanpath_id += 1

            temple_gaze = temple_gaze[:scanpath_id]


            gaze_ = np.zeros((temple_gaze.shape[0],max,3))
            for scanpath_id in range(0,temple_gaze.shape[0]):
                gaze_[scanpath_id] = suppor_lib.sphere2xyz(torch.tensor(temple_gaze[scanpath_id]))
                self.info['train']['num_scanpath'] += 1
                scan_length.append(scanpath_length[scanpath_id])
            dic = {"image": image, "scanpaths": gaze_}
            twoDict(self.image_and_scanpath_dict, "train",
                    file_name[:-4],
                    dic)
            image_id += 1

        self.info['train']['num_image'] = image_id
        self.info['train']['scanpath_length'] = scan_length
        self.info['train']['max_scan_length'] = max




    def run(self):
        ''
        ' PATH PREPARE '
        print(self.images_path)
        # for file_name in os.listdir(self.images_path):
        #     if ".png" in file_name:
        #         if file_name in self.test_set:
        #             self.images_test_list.append(os.path.join(self.images_path, file_name))
        #         else:
        #             self.images_train_list.append(os.path.join(self.images_path, file_name))


        ' GET TRAINING SET '
        print('\nProcessing [Training Set]\n')
        self.get_train_set()

        # ' GET TEST SET '
        # print('\nProcessing [Test Set]\n')
        # self.get_test_set()

        ' RECORD DATABASE INFORMATION '
        self.image_and_scanpath_dict['info'] = self.info

        return self.image_and_scanpath_dict


if __name__ == '__main__':

    Datasets = ['AOI']

    for dataset in Datasets:
        forward(dataset)

#org_img_y_x = lat_lon / np.array([180.0, 360.0]) * np.array(org_height_width)