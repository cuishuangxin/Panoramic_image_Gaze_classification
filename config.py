import torch

DATABASE_ROOT = 'Datasets'

dic_Sitzmann = {
    'IMG_PATH': DATABASE_ROOT + '/Sitzmann/rotation_imgs/imgs/',
    'GAZE_PATH': DATABASE_ROOT + '/Sitzmann/vr/',
    # 'TEST_SET': ['cubemap_0000.png', 'cubemap_0006.png', 'cubemap_0009.png']
    'TEST_SET': []
}


'Parameters for training'
image_size = [128, 256]
seed = 1234
num_epochs = 400
learning_rate = 0.00003
lr_decay = 0.99998
weight_decay = 2.0
mini_batch_size = 64
annealing_epochs = 10
minimum_annealing_factor = 0.2
load_model = None
load_opt = None
# save_root = './model1+/'
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


