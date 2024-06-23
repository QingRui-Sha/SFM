import os
import sys
import glob
import numpy as np
import time
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import torch
import nibabel as nib
def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img
class Dataset_demo(Data.Dataset):
    def __init__(self, train_files):
                #  image_file_dir,seg_file_dir=None,inverse_kernel_file_dir=None,target_coordinate_repr_file_dir=None,dynamic_coor_file_dir=None,key_point_file_dir=None,boundary_vessels_file_dir=None,
                #  prefix_image=None,prefix_seg=None,prefix_inverse_kernel=None,prefix_target_coordinate_repr=None,prefix_dynamic_coor=None,prefix_key_point=None,prefix_boundary_vessels=None,w_flow_mse=0):
        super(Dataset_demo, self).__init__()
        self.image_filenames =train_files
        
    def __getitem__(self, index):
        index=int(index/2)

        image = nib.load(self.image_filenames[index*2])
        image =image.get_fdata().squeeze()
        image = image[np.newaxis, :, :, :]
        
        image_fixed = nib.load(self.image_filenames[index*2+1])
        image_fixed =image_fixed.get_fdata().squeeze()
        image_fixed = image_fixed[np.newaxis, :, :, :]

        return torch.from_numpy(imgnorm(image)).float(), torch.from_numpy(imgnorm(image_fixed)).float()
    def __len__(self):
        return len(self.image_filenames)
    
