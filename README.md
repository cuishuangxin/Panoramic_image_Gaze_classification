# Make Gaze Data Great Again for Panoramic Scenes: A Progressive Iterative Self-Paced Learning


1.Dual-predictor

# Training  
1. To reproduce the training and validation dataset, please referring to data_process.py.
2. Execute:  
```
python train.py --seed=1234 --dataset='./Datasets/AOI.pkl' --lr=0.0003 --bs=64 --epochs=400 --save_root='./model/'
```
# Testing
```
python inference.py --model='./model/model_lr-0.0003_bs-64_epoch-217.pkl' --inDir='./demo/input' --outDir='./demo/output' --n_scanpaths=20 --length=15 --if_plot=True
```  

2.classify
# split
1. Perform one-dimensional linear clustering on the data, please referring to slic.py.
2. According to the predicted results, the trajectory similarity is calculated with the test set to determine the viewpoint type.
