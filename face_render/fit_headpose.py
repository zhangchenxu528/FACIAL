import os, sys
from os.path import join, exists, abspath, dirname
import numpy as np
from time import time
import matplotlib.pyplot as plt
import cv2
import face3d
from face3d import mesh
from face3d.morphable_model.fit import estimate_expression, estimate_shape, fit_points, fit_points_for_show
from load_data import BFM 
from scipy.io import loadmat,savemat
import pandas as pd
from scipy.signal import savgol_filter
import argparse

parser = argparse.ArgumentParser(description='fit_headpose_setting')
parser.add_argument('--csv_path', type=str, default='/content/FACIAL/video_preprocess/train1_openface/train1_512_audio.csv')
parser.add_argument('--deepface_path', type=str, default='/content/FACIAL/video_preprocess/train1_deep3Dface/train1.npz')
parser.add_argument('--save_path', type=str, default='/content/FACIAL/video_preprocess/train1_posenew.npz')
opt = parser.parse_args()


# --- 1. load model
facemodel = BFM()
n_exp_para = facemodel.exBase.shape[1]

kpt_ind = facemodel.keypoints
triangles = facemodel.tri


csv_path = opt.csv_path
csvinfo=pd.read_csv(csv_path)
num_image = len(csvinfo)
base = int(csvinfo.iloc[0]['frame'])-1
deepface_path = opt.deepface_path
save_path = opt.save_path


realparams = np.load(open(deepface_path, 'rb'))
realparams = realparams['face']

idparams = realparams[0,0:80]
texparams = realparams[0,144:224]
gammaparams = realparams[0,227:254]

h = 512
w = 512

headpose = np.zeros((num_image,258),dtype=np.float32)
base = int(csvinfo.iloc[0]['frame'])-1
# --- 2. fit head pose for each frame
for frame_count in range(1,num_image+1):
    if frame_count % 1000 == 0:
        print(frame_count)
    subcsvinfo = csvinfo[csvinfo['frame']==frame_count+base]
    x = np.zeros((68,2),dtype=np.float32)
    for i in range(68):
        x[i,0] = subcsvinfo.iloc[0][' x_'+str(i)]-w/2
        x[i,1] = (h-subcsvinfo.iloc[0][' y_'+str(i)])- h/2 -1
    X_ind = kpt_ind

    fitted_sp, fitted_ep, fitted_s, fitted_R, fitted_t = fit_points(x, X_ind, facemodel, np.expand_dims(idparams,0), n_ep = n_exp_para, max_iter = 10)

    fitted_angles = mesh.transform.matrix2angle(fitted_R)
    fitted_angles = np.array([fitted_angles])

    chi_prev = np.concatenate((fitted_angles[0,:],[fitted_s],fitted_t,realparams[frame_count-1,80:144]),axis=0)
    params = np.concatenate((chi_prev,idparams,texparams,gammaparams),axis=0)
    headpose[frame_count-1,:] = params
# additional smooth
headpose1 = np.zeros((num_image,258),dtype=np.float32)
headpose1 = savgol_filter(headpose, 5, 3, axis=0)

np.savez(save_path, face = headpose1)
