# UDAS2S
a novel unsupervised domain adaptive method based on adversarial learning and semi-supervised learning for unpaired cross-modality image segmentation

# Environment and Dependencies
Python 3.9
Pytorch 1.11.0
scipy
scikit-image


# Daataset
The original dataset can be obtained from the following link:
https://github.com/FupingWu90/CT_MR_2D_Dataset_DA.git

After obtaining the original dataset, the dataset can be preprocessed using the Dataset/ni2npy.Py file.
## train_MRI.txt contains the .png file names with content of
img10_slice1.npy.png  
img10_slice2.npy.png  
img10_slice3.npy.png  
img10_slice4.npy.png  
...
img10_sliceN.npy.png  

## train_CT.txt contains the .png file names with content of
img10_slice1.npy.png  
img10_slice2.npy.png  
img10_slice3.npy.png  
img10_slice4.npy.png  
...
img10_sliceN.npy.png  
  
# Train and Test
Save the preprocessed data of nii2Npy.Py as PNG images, and divide the processed MRI and CT into training and testing sets. Just modify the '-- rawA_Adir', '-- rawA_set_gdir', '-- raw_S-DIr', and '-- raw_Set_gdir' to your correct path in the Code/options/base_options. py file. You can run the train.exe file to train the model.
Call the trained. pth model in test.py to test.

Special thanks to the article 'Anatomy Constrained Contrastive Learning for Synthetic Segmentation without Ground truth' for its help.
https://github.com/bbbbbbzhou/AccSeg-Net.git

# Contact
If you have any question, please file an issue or contact the author:
E-mail: xyy_20152451@163.com


