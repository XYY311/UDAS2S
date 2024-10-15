


import time
import os
import sublist
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from util.visualizer import Visualizer
from models import networks
import torch
from scipy.ndimage.interpolation import zoom
import numpy as np
from medpy import metric
import cv2
import scipy.io as sio

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = 'cuda'

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    pred = pred.astype(int)
    gt = gt.astype(int)

    if (pred.sum()) & (gt.sum()) > 0:
        dice = metric.binary.dc(pred, gt)
        # iou = binary_iou(pred, gt)
        asd = metric.binary.asd(pred, gt)
        ji = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, asd, hd95# ji
    else:
        return 0,0, 0#, 0

# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#
#     dice = metric.binary.dc(pred, gt)
#     asd = metric.binary.asd(pred, gt)
#     hd95 = metric.binary.hd95(pred, gt)
#     return dice,asd,hd95



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    seg_output_dir = opt.test_seg_output_dir
    test_img_list_file = opt.test_img_list_file
    opt.imglist_testB = sublist.dir2list(opt.test_B_dir, test_img_list_file)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#test images = %d' % dataset_size)

    model = networks.define_S(opt.input_nc, opt.output_nc_seg*5, opt.ngf, opt.netS, opt.normS, not opt.no_dropout,
                              opt.init_type, opt.init_gain, opt.no_atialias, opt.no_antialias_up, opt.gpu_ids, opt)


    save_mode_path = 'G:\XYY\Cross_modality_Seg\MMWHS对比模型\AccSeg-Net-main - MMWHS\checkpoints\CBCT\experiment_cut2seg\latest_net_D.pth'
    model.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    model.eval()
    # 获取模型参数总数量
    total_params = sum(p.numel() for p in model.parameters())

    # 获取模型内存大小 (单位：MB)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # 单个参数的字节大小

    model_size_mb = param_size / (1024 ** 2)  # 转换为MB
    print(f"Total parameters: {total_params}")
    print(f"Model size: {model_size_mb:.2f} MB")
    #
    # model = create_model(opt)      # create a model given opt.model and other options
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    # dice_MYO = 0
    # dice_LV = 0
    # dice_LA = 0
    # dice_RV = 0
    # dice_RA = 0
    dice_MYO = []
    dice_LV = []
    dice_LA = []
    dice_RV = []
    dice_RA = []
    assd_MYO = []
    assd_LV = []
    assd_LA = []
    assd_RV = []
    assd_RA = []
    hd_MYO = []
    hd_LV = []
    hd_LA = []
    hd_RV = []
    hd_RA = []
    images = np.zeros((160,192,192))
    labels = np.zeros((160,5,192,192))
    Predications = np.zeros((160,5,192,192))
    for i, data in enumerate(dataset):
        img = data['B'].to(device)
        lab = data['Seg']
        image = torch.squeeze(img)
        images[i,...] = image.data.cpu().numpy()

        label = torch.squeeze(lab)
        label = label.data.cpu().numpy()
        labels[i, ...] = label
        # np.savetxt('lab.txt', np.squeeze(lab.cpu().numpy()), fmt='%d', delimiter=',')

        with torch.no_grad():
            start_time = time.time()
            output = model(img)
            end_time = time.time()
            total_time = end_time - start_time

            print(f"Total inference time on test dataset: {total_time:.2f} seconds")

            # output = torch.softmax(output, dim=1)
            out = torch.squeeze(output)
            out = out.data.cpu().detach().numpy()
            # np.savetxt('out.txt',out, fmt='%d', delimiter=',')
        #  MYO==1 LV==2 LA==3 RV==4  RA==5
        Predications[i,...] = out
        out_MYO = out[0,...]
        lab_MYO = label[0,...]
        out_MYO[out_MYO > 0.5] = 1
        out_MYO[out_MYO < 0.5] = 0

        out_LV = out[1, ...]
        lab_LV = label[1, ...]
        out_LV[out_LV > 0.5] = 1
        out_LV[out_LV < 0.5] = 0

        out_LA = out[2, ...]
        lab_LA = label[2, ...]
        out_LA[out_LA > 0.5] = 1
        out_LA[out_LA < 0.5] = 0

        out_RV = out[3, ...]
        lab_RV = label[3, ...]
        out_RV[out_RV > 0.5] = 1
        out_RV[out_RV < 0.5] = 0

        out_RA = out[4, ...]
        lab_RA = label[4, ...]
        out_RA[out_RA > 0.5] = 1
        out_RA[out_RA <0.5 ] = 0
        # m = out_MYO.sum()


        d_MYO,a_MYO,c_MYO = calculate_metric_percase(out_MYO, lab_MYO)
        dice_MYO.append(d_MYO)
        assd_MYO.append(a_MYO)
        hd_MYO.append(c_MYO)
        d_LV,a_LV,c_LV = calculate_metric_percase(out_LV, lab_LV)
        dice_LV.append(d_LV)
        assd_LV.append(a_LV)
        hd_LV.append(c_LV)
        d_LA,a_LA,c_LA = calculate_metric_percase(out_LA, lab_LA)
        dice_LA.append(d_LA)
        assd_LA.append(a_LA)
        hd_LA.append(c_LA)
        d_RV,a_RV,c_RV= calculate_metric_percase(out_RV, lab_RV)
        dice_RV.append(d_RV)
        assd_RV.append(a_RV)
        hd_RV.append(c_RV)
        d_RA,a_RA,c_RA = calculate_metric_percase(out_RA, lab_RA)
        dice_RA.append(d_RA)
        assd_RA.append(a_RA)
        hd_RA.append(c_RA)




        image = image.data.cpu().detach().numpy()* 0.5 + 0.5
        # label = label.data.cpu().detach().numpy()*50
        # imgSave = np.concatenate([image, label, out], 1)
        imgSave = np.concatenate([image, lab_MYO,lab_LV,lab_LA,lab_RV,lab_RA, out_MYO,out_LV,out_LA,out_RV,out_RA], 1)
        savePath = os.path.join(opt.testImgSaveDir, '%03d_source_target.png' % i)
        cv2.imwrite(savePath, (imgSave * 255).astype(np.uint8))

    #
    #     # print('test result:', dice_MYO / 160, dice_LV / 160, dice_LA / 160, dice_RV / 160, dice_RA / 160)
    # 创建一个字典，将变量名作为键，变量本身作为值
    variables = {'img': images, 'lab': labels, 'pre': Predications}

    # 保存到MAT文件
    sio.savemat('AccSeg_img_lab_preCT2MRI.mat', variables)
    print('test result-dice:', np.mean(dice_MYO), np.mean(dice_LV),np.mean(dice_RV),np.mean(dice_LA),np.mean(dice_RA))
    # print('test result-std:', np.std(dice_MYO), np.std(dice_LV),np.std(dice_RV),np.std(dice_LA),np.std(dice_RA))
    print('test result-assd:', np.mean(assd_MYO), np.mean(assd_LV), np.mean(assd_RV), np.mean(assd_LA),np.mean(assd_RA))
    print('test result-hd95:', np.mean(hd_MYO), np.mean(hd_LV), np.mean(hd_RV), np.mean(hd_LA),np.mean(hd_RA))
    # print('test result-std:', np.std(assd_MYO), np.std(assd_LV), np.std(assd_RV), np.std(assd_LA), np.std(assd_RA))
        # model.set_input(data)
        # model.test()
        # visuals = model.get_current_visuals()
        # img_path = model.get_image_paths()
        # print('processing image... %s' % img_path)
        # visualizer.save_seg_images_to_dir(seg_output_dir, visuals, img_path)
