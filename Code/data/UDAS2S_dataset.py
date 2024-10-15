import os.path

import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import cv2
import data.random_pair as random_pair


class TrainDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.dir_A = opt.raw_A_dir
        self.dir_B = opt.raw_B_dir
        self.dir_Seg = opt.raw_A_seg_dir

        self.A_paths = opt.imglist_A
        self.B_paths = opt.imglist_B

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if not self.opt.isTrain:
            self.skipcrop = True
            self.skiprotate = True
        else:
            self.skipcrop = False
            self.skiprotate = False

        if self.skipcrop:
            osize = [opt.crop_size, opt.crop_size]
        else:
            osize = [opt.load_size, opt.load_size]

        if self.skiprotate:
            angle = 0
        else:
            angle = opt.angle

        transform_list = []
        transform_list.append(random_pair.randomrotate_pair(angle))    # scale the image
        self.transforms_rotate = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))    # scale the image
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Resize(osize, Image.NEAREST))    # scale the segmentation
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_pair.randomcrop_pair(opt.crop_size))    # random crop image & segmentation
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        self.transforms_normalize = transforms.Compose(transform_list)

    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = os.path.join(self.dir_A, self.A_paths[index_A])
        Seg_path = os.path.join(self.dir_A, self.A_paths[index_A])
        Seg_path = Seg_path.replace('.png', '_label.png') #'_mask.png'

        # index_B = random.randint(0, self.B_size - 1)
        index_B = index_A
        B_path = os.path.join(self.dir_B, self.B_paths[index_B])

        A_img = Image.open(A_path).convert('L')
        Seg_img = Image.open(Seg_path).convert('I')  #当标签为1,2,3,4,5不用他将会全为0
        # Seg_img = cv2.imread(Seg_path,0)
        # Seg_img = Image.fromarray(Seg_img)
        B_img = Image.open(B_path).convert('L')

        A_img = self.transforms_scale(A_img)
        B_img = self.transforms_scale(B_img)
        Seg_img = self.transforms_seg_scale(Seg_img)

        if not self.skiprotate:
            [A_img, Seg_img] = self.transforms_rotate([A_img, Seg_img])
            [B_img] = self.transforms_rotate([B_img])

        if not self.skipcrop:
            [A_img, Seg_img] = self.transforms_crop([A_img, Seg_img])
            [B_img] = self.transforms_crop([B_img])

        A_img = self.transforms_toTensor(A_img)
        B_img = self.transforms_toTensor(B_img)
        Seg_img = self.transforms_toTensor(Seg_img)

        A_img = self.transforms_normalize(A_img)
        B_img = self.transforms_normalize(B_img)

        # Seg_img[Seg_img > 0] = 1
        # Seg_img[Seg_img < 1] = 0

        Seg_imgs = torch.Tensor(self.opt.output_nc_seg*5, self.opt.crop_size, self.opt.crop_size)
        Seg_img = torch.squeeze(Seg_img).data.cpu().numpy()
        # N,_,_ = Seg_imgs.shape()
        Seg_1= Seg_img.copy()
        # np.savetxt('Sge1.txt', Seg_1, fmt='%d', delimiter=',')
        Seg_2= Seg_img.copy()
        # np.savetxt('Sge2.txt', Seg_2, fmt='%d', delimiter=',')
        Seg_3= Seg_img.copy()
        Seg_4= Seg_img.copy()
        Seg_5= Seg_img.copy()
        Seg_1[Seg_1!=1]=0
        # np.savetxt('Sge11.txt', Seg_1, fmt='%d', delimiter=',')
        # Seg_2 = [0 if num != 2 else 1 for num in Seg_2]
        Seg_2[Seg_2 !=2] = 0
        Seg_2[Seg_2 ==2]=1
        # np.savetxt('Sge222.txt', Seg_2, fmt='%d', delimiter=',')
        Seg_3[Seg_3!=3]=0
        Seg_3[Seg_3==3] = 1
        Seg_4[Seg_4!=4]=0
        Seg_4[Seg_4 == 4] = 1
        Seg_5[Seg_5!=5]=0
        Seg_5[Seg_5 == 5] = 1
        # np.savetxt('Sge5.txt', Seg_5, fmt='%d', delimiter=',')

        #
        Seg_imgs[0,:,:] = torch.Tensor(Seg_1)
        Seg_imgs[1,:,:] = torch.Tensor(Seg_2)
        Seg_imgs[2,:,:] = torch.Tensor(Seg_3)
        Seg_imgs[3,:,:] = torch.Tensor(Seg_4)
        Seg_imgs[4,:,:] = torch.Tensor(Seg_5)
        # np.savetxt('Sge1.txt', Seg_1.data.cpu().numpy(), fmt='%d', delimiter=',')
        # np.savetxt('Sge2.txt', Seg_2.data.cpu().numpy(), fmt='%d', delimiter=',')
        # np.savetxt('Sge3.txt', Seg_3.data.cpu().numpy(), fmt='%d', delimiter=',')
        # np.savetxt('Sge4.txt', Seg_4.data.cpu().numpy(), fmt='%d', delimiter=',')
        # np.savetxt('Sge5.txt', Seg_5.data.cpu().numpy(), fmt='%d', delimiter=',')


        return {'A': A_img, 'B': B_img, 'Seg': Seg_imgs, 'Seg_one': Seg_img,
                'A_paths': A_path, 'B_paths': B_path, 'Seg_paths': Seg_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'TrainDataset'


class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_B = opt.test_B_dir

        self.B_filenames = opt.imglist_testB
        # self.B_filenames = opt.sub_list_B
        self.B_size = len(self.B_filenames)

        if not self.opt.isTrain:
            self.skipcrop = True
            self.skiprotate = True
        else:
            self.skipcrop = False
            self.skiprotate = False

        if self.skipcrop:
            osize = [opt.crop_size, opt.crop_size]
        else:
            osize = [opt.load_size, opt.load_size]

        if self.skiprotate:
            angle = 0
        else:
            angle = opt.angle

        transform_list = []
        transform_list.append(random_pair.randomrotate_pair(angle))    # scale the image
        self.transforms_rotate = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))    # scale the image
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Resize(osize, Image.NEAREST))    # scale the segmentation
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_pair.randomcrop_pair(opt.crop_size))    # random crop image & segmentation
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        self.transforms_normalize = transforms.Compose(transform_list)

    def __getitem__(self, index):
        B_filename = self.B_filenames[index % self.B_size]
        B_path = os.path.join(self.dir_B, B_filename)
        B_img = Image.open(B_path).convert('L')
        B_img = self.transforms_scale(B_img)
        B_img = self.transforms_toTensor(B_img)
        B_img = self.transforms_normalize(B_img)

        Seg_filename = self.B_filenames[index % self.B_size]
        Seg_path = os.path.join(self.dir_B, Seg_filename)
        Seg_path = Seg_path.replace('.png', '_label.png')
        Seg_img = Image.open(Seg_path).convert('I')
        # Seg_img = cv2.imread(Seg_path, 0)
        # Seg_img = Image.fromarray(Seg_img)
        Seg_img = self.transforms_toTensor(Seg_img)

        # Seg_img[Seg_img > 0] = 1
        # Seg_img[Seg_img < 1] = 0
        #
        # Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.crop_size, self.opt.crop_size)
        # Seg_imgs[0, :, :] = Seg_img #== 1
        Seg_imgs = torch.Tensor(self.opt.output_nc_seg * 5, self.opt.crop_size, self.opt.crop_size)
        Seg_img = torch.squeeze(Seg_img).data.cpu().numpy()
        # N,_,_ = Seg_imgs.shape()
        Seg_1 = Seg_img.copy()
        # np.savetxt('Sge1.txt', Seg_1, fmt='%d', delimiter=',')
        Seg_2 = Seg_img.copy()
        # np.savetxt('Sge2.txt', Seg_2, fmt='%d', delimiter=',')
        Seg_3 = Seg_img.copy()
        Seg_4 = Seg_img.copy()
        Seg_5 = Seg_img.copy()
        Seg_1[Seg_1 != 1] = 0
        # np.savetxt('Sge11.txt', Seg_1, fmt='%d', delimiter=',')
        # Seg_2 = [0 if num != 2 else 1 for num in Seg_2]
        Seg_2[Seg_2 != 2] = 0
        Seg_2[Seg_2 == 2] = 1
        # np.savetxt('Sge222.txt', Seg_2, fmt='%d', delimiter=',')
        Seg_3[Seg_3 != 3] = 0
        Seg_3[Seg_3 == 3] = 1
        Seg_4[Seg_4 != 4] = 0
        Seg_4[Seg_4 == 4] = 1
        Seg_5[Seg_5 != 5] = 0
        Seg_5[Seg_5 == 5] = 1
        # np.savetxt('Sge5.txt', Seg_5, fmt='%d', delimiter=',')

        #
        Seg_imgs[0, :, :] = torch.Tensor(Seg_1)
        Seg_imgs[1, :, :] = torch.Tensor(Seg_2)
        Seg_imgs[2, :, :] = torch.Tensor(Seg_3)
        Seg_imgs[3, :, :] = torch.Tensor(Seg_4)
        Seg_imgs[4, :, :] = torch.Tensor(Seg_5)

        return {'B': B_img, 'Seg': Seg_imgs,
                'B_paths': B_path, 'Seg_paths': Seg_path}

    def __len__(self):
        return self.B_size

    def name(self):
        return 'TestDataset'