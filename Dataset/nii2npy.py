import os
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

input_folder = './CT_woGT/imgs'
output_folder = './CT_woGT/npy/imgs'
# data = np.load('CHAOS1_0.npy')
# label = np.load('CHAOS1_1.npy')
# a= np.max(label)

def noralization(img):
    #use non-zero ROI mean and std to normalize
    # mask = img.copy()
    # mask[img>0] = 1
    # mean = np.sum(mask*img) / np.sum(mask)
    # std = np.sqrt(np.sum(mask * (img - mean)**2) / np.sum(mask))
    # img = (img-mean)/std
    img_min = np.min(img)
    img_max = np.max(img)
    img = (img-img_min)/(img_max-img_min)
    return img

for filename in os.listdir(input_folder):
    if filename.endswith('.nii.gz'):
    
        img = nib.load(os.path.join(input_folder, filename))
      
        data = img.get_fdata()
        data = np.squeeze(data)
        data = noralization(data)

        # data[data==421] = 0
        # data[(data != 205) & (data != 500) & (data != 420) & (data != 600) & (data != 550)] = 0
       
        # data[data == 500] = 100
        # data[data == 600] = 200
        # data = np.interp(data, (data.min(), data.max()), (0, 255)).astype(np.uint8)
    
        # data[data == 205] = 1
        # data[data == 500] = 2
        # data[data == 420] = 3
        # data[data == 600] = 4
        # data[data == 550] = 5

        # 【240,220]-->[192,192]
      
        left = (240 - 192) // 2
        top = (220 - 192) // 2
        data = data[left:left + 192, top:top + 192]
        m = np.max(data)

   
        np.save(os.path.join(output_folder,'img'+ filename[3:-7] + '.npy'), data)
