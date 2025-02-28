import time

import torch
from driver.base_train_helper import BaseTrainHelper
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from fistula_data.utils import (CenterCrop, RandomCrop,
                                RandomRotFlip, ToTensor)
from fistula_data.XiTSDataLoader import XiTS, XiWSTS
import math
from tqdm import tqdm
import h5py
import SimpleITK as sitk
import torch
from driver.base_train_helper import BaseTrainHelper
from torch.utils.data import DataLoader
import numpy as np
import cv2
from os.path import exists, join
from einops import rearrange, reduce, repeat
from commons.utils import *
from torch.cuda import empty_cache
from torch.nn.utils import clip_grad_norm_
from commons.evaluation import compute_all_metric_for_single_seg
from skimage.transform import resize
import matplotlib
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    RandGaussianNoised,
    RandBiasFieldd,
    RandGibbsNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandFlipd,
    RandCoarseDropoutd,
    RandRotated,
    RandZoomd,
    RandAffined,
    RandSpatialCropd,
    ScaleIntensityRanged,
    RandCropByLabelClassesd,
    EnsureChannelFirstd,

)

plt.rcParams.update({'figure.max_open_warning': 20})


class XiTSHelper(BaseTrainHelper):
    def __init__(self, criterions, config):
        super(XiTSHelper, self).__init__(criterions, config)

    def merge_batch(self, batch):
        image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
        image_segment = [torch.unsqueeze(inst["image_segment"], dim=0) for inst in batch]
        image_patch = torch.cat(image_patch, dim=0)
        image_segment = torch.cat(image_segment, dim=0)
        return {"image_patch": image_patch,
                "image_segment": image_segment,
                }

    def read_data(self, dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                segment = batch['image_segment']
                image = image.to(self.equipment).float()
                segment = segment.to(self.equipment).long()
                yield {
                    "image_patch": image,
                    "image_segment": segment,
                }

    def get_train_loader(self, seed=1337):
        from torchvision import transforms
        image_list = sorted(glob(join(self.config.data_path, '*.npy')), reverse=False)
        np.random.seed(seed)
        np.random.shuffle(image_list)
        if self.config.data_name == 'LiTS':
            # 118 total
            vali_num = 8
            test_num = 10
        elif self.config.data_name == 'KiTS':
            vali_num = 20
            test_num = 40
        elif self.config.data_name == 'MSD':
            vali_num = 8
            test_num = 10
        else:
            raise ValueError("data set not defined")
        tr_image_list = image_list[:len(image_list) - test_num]
        val_image_list = tr_image_list[len(tr_image_list) - vali_num:]
        tr_image_list = tr_image_list[:len(tr_image_list) - vali_num]
        train_size = self.config.default_seg_label_size
        tr_training = tr_image_list[:train_size]
        tr_training_un = tr_image_list[train_size:]
        train_transforms = Compose(
            [
                EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    ratios=[0.3, 0.7],
                    spatial_size=[self.config.patch_x, self.config.patch_y, self.config.patch_z],
                    num_classes=2,
                    num_samples=1,
                    allow_smaller=True
                ),
                ToTensord(keys=['image', 'label'])
            ]
        )

        db_train = XiTS(tr_training, base_dir=self.config.data_path,
                        transform=train_transforms, input_size=(self.config.patch_x, self.config.patch_y,
                                                                self.config.patch_z))
        train_dataloader = DataLoader(db_train, shuffle=True, collate_fn=self.merge_batch,
                                      batch_size=self.config.train_seg_batch_size, drop_last=True,
                                      num_workers=self.config.workers, pin_memory=True)
        val_dataset = XiTS(val_image_list, base_dir=self.config.data_path,
                           transform=None)
        if len(tr_training_un) > 0:
            train_dataset_un = XiTS(tr_training_un, base_dir=self.config.data_path,
                                    transform=train_transforms, input_size=(self.config.patch_x, self.config.patch_y,
                                                                            self.config.patch_z))
            train_dataloader_un = DataLoader(train_dataset_un, shuffle=True, collate_fn=self.merge_batch,
                                             batch_size=self.config.train_seg_batch_size, drop_last=True,
                                             num_workers=self.config.workers, pin_memory=True)
        else:
            train_dataloader_un = None

        return train_dataloader, train_dataloader_un, val_dataset.data

    def get_test_data_loader(self, seed=1337):
        image_list = sorted(glob(join(self.config.data_path, '*.npy')), reverse=False)
        np.random.seed(seed)
        np.random.shuffle(image_list)
        if self.config.data_name == 'LiTS':
            # 118 total
            test_num = 10
        elif self.config.data_name == 'KiTS':
            test_num = 40
        elif self.config.data_name == 'MSD':
            test_num = 10
        else:
            raise ValueError("data set not defined")
        te_image_list = image_list[len(image_list) - test_num:]
        test_dataset = XiTS(te_image_list, base_dir=self.config.data_path,
                            transform=None)
        return test_dataset.data

    def test_single_case(self, net, image, stride_xy, stride_z, patch_size, num_classes=1, image_name=''):
        w, h, d = image.shape

        # if the size of image is less than patch_size, then padding it
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0] - w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1] - h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2] - d
            add_pad = True
        else:
            d_pad = 0
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
        if add_pad:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                   (dl_pad, dr_pad)], mode='constant', constant_values=0)
        ww, hh, dd = image.shape

        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        # print("{}, {}, {}".format(sx, sy, sz))
        score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
        cnt = np.zeros(image.shape).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image[xs:xs + patch_size[0],
                                 ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(
                        test_patch, axis=0), axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).to(self.device)

                    with torch.no_grad():
                        # FOR MCF
                        if isinstance(net, list):
                            y1 = net[0](test_patch)
                            y2 = net[1](test_patch)
                            prob_maps = torch.softmax(y1, dim=1)
                            prob_maps_r = torch.softmax(y2, dim=1)
                            y_p = (prob_maps + prob_maps_r) / 2
                        elif isinstance(net, tuple):
                            # FOR MCF
                            y1 = net[0](test_patch)
                            y2 = net[1](test_patch)
                            prob_maps = torch.softmax(y1, dim=1)
                            prob_maps_r = torch.softmax(y2, dim=1)
                            y_p = (prob_maps + prob_maps_r) / 2
                        else:
                            y1 = net(test_patch)
                            if self.config.model == 'asnet':
                                patch_test_mask = torch.sigmoid(y1)
                                y_p = torch.cat([1 - patch_test_mask, patch_test_mask], dim=1)
                            else:
                                # FOR CCT
                                if isinstance(y1, list):
                                    y1 = y1[0]
                                    y_p = torch.softmax(y1, dim=1)
                                elif isinstance(y1, tuple):
                                    y1 = y1[0]
                                    y_p = torch.softmax(y1, dim=1)
                                else:
                                    y_p = torch.softmax(y1, dim=1)
                    # visual_batch(test_patch, self.config.tmp_dir, 'images_%s_%d_%d_%d_raw' % (image_name, x, y, z),
                    #              channel=1,
                    #              nrow=16)
                    # visual_batch(y1[:, 1, :, :, :].unsqueeze(1), self.config.tmp_dir,
                    #              'images_%s_%d_%d_%d_pred' % (image_name, x, y, z),
                    #              channel=1,
                    #              nrow=16)
                    y_p = y_p.cpu().data.numpy()
                    y_p = y_p[0, :, :, :, :]
                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y_p
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
        score_map = score_map / np.expand_dims(cnt, axis=0)
        label_map = np.argmax(score_map, axis=0)
        # visual_batch(torch.from_numpy(score_map[1, :, :, :]).unsqueeze(0).unsqueeze(0), self.config.tmp_dir,
        #              'images_%s_%d_%d_%d_pred_label' % (image_name, x, y, z),
        #              channel=1,
        #              nrow=16)
        if add_pad:
            label_map = label_map[wl_pad:wl_pad + w,
                        hl_pad:hl_pad + h, dl_pad:dl_pad + d]
            score_map = score_map[:, wl_pad:wl_pad +
                                            w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        return label_map, score_map

    def test_all_case(self, test_data, pmodel, num_classes=4, patch_size=(48, 160, 160),
                      stride_xy=32,
                      stride_z=24, isTest=False):
        fistula_segmentation_scores = {}
        fistula_segmentation_metrics = {}
        total_metric = np.zeros((num_classes - 1, 2))
        print("Validation begin  %.4f" % (time.time()))
        if isTest:
            output_dir = join(self.config.tmp_dir, 'infer')
            mkdir_if_not_exist([output_dir])
        for image_path in test_data:
            # tt = time.time()
            # print("Read begin  %.4f" % (tt))
            ids = basename(image_path)[:-4]
            img = np.load(image_path)
            image = img[..., 0].astype(np.float32)
            label = img[..., 1].astype(np.float32)
            # print("Read end  %.4f" % (time.time() - tt))
            # tt = time.time()
            # print("Test begin  %.4f" % (tt))
            prediction, score_map = self.test_single_case(pmodel,
                                                          image, stride_xy, stride_z, patch_size,
                                                          num_classes=num_classes,
                                                          image_name=ids)
            # print("Test end  %.4f" % (time.time() - tt))
            # tt = time.time()
            # print("Cal begin  %.4f" % (tt))
            if isTest:
                scores = compute_all_metric_for_single_seg(label, prediction)
                for metric in scores:
                    if metric not in fistula_segmentation_scores:
                        fistula_segmentation_scores[metric] = []
                    fistula_segmentation_scores[metric].extend(scores[metric])
                # print("Cal end  %.4f" % (time.time() - tt))

                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, output_dir +
                                "/{}_pred.nii.gz".format(ids))

                img_itk = sitk.GetImageFromArray(image)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, output_dir +
                                "/{}_img.nii.gz".format(ids))

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, output_dir +
                                "/{}_lab.nii.gz".format(ids))
                visual_batch(torch.from_numpy(image).unsqueeze(0).unsqueeze(0), self.config.tmp_dir,
                             'images_%s_raw' % (ids),
                             channel=1,
                             nrow=16)
                visual_batch(torch.from_numpy(label).unsqueeze(0).unsqueeze(0), self.config.tmp_dir,
                             'images_%s_label' % (ids),
                             channel=1,
                             nrow=16)
                visual_batch(torch.from_numpy(score_map[self.config.classes - 1]).unsqueeze(0).unsqueeze(0),
                             self.config.tmp_dir,
                             'images_%s_prob' % (ids),
                             channel=1,
                             nrow=16)
                visual_batch(torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0), self.config.tmp_dir,
                             'images_%s_pred' % (ids),
                             channel=1,
                             nrow=16)
            else:
                for i in range(1, num_classes):
                    total_metric[i - 1, :] += self.cal_metric(label == i, prediction == i)
        if isTest:
            info = ''
            metrics = ''
            for m in fistula_segmentation_scores:
                fistula_segmentation_metrics[m] = np.mean(fistula_segmentation_scores[m])
                metrics += (m + ' \t')
                info += ('{val:.5f}  \t'.format(val=fistula_segmentation_metrics[m]))
            print(metrics)
            print(info)
            seg_dice = np.mean(fistula_segmentation_scores['dice'])
            zipDir(output_dir, output_dir + '.zip')
            shutil.rmtree(output_dir)
        else:
            seg_dice = total_metric[0, 0] / len(test_data)
            print("Valid: Dice {}, HD {}".format(total_metric[0, 0] / len(test_data),
                                                 total_metric[0, 1] / len(test_data)))

        print("Validation end Dice %.5f, Time %.4f" % (seg_dice, time.time()))
        return seg_dice, fistula_segmentation_metrics

    def predict_whole(self, pmodel, test_data, vis=False, isTest=False):
        seg_acc, fistula_segmentation_metrics = self.test_all_case(test_data, pmodel, num_classes=self.config.classes,
                                                                   patch_size=(self.config.patch_x, self.config.patch_y,
                                                                               self.config.patch_z),
                                                                   stride_xy=int(self.config.patch_x // 3 * 2),
                                                                   stride_z=int(self.config.patch_z // 3 * 2),
                                                                   isTest=isTest)
        return {
            'vali/seg_acc': seg_acc,
            'pred_acc': fistula_segmentation_metrics
        }


    def get_ws_train_loader(self, seed=1337):
        from torchvision import transforms
        image_list = sorted(glob(join(self.config.data_path, '*.npy')), reverse=False)
        np.random.seed(seed)
        np.random.shuffle(image_list)
        if self.config.data_name == 'LiTS':
            # 118 total
            vali_num = 8
            test_num = 10
        elif self.config.data_name == 'KiTS':
            vali_num = 20
            test_num = 40
        elif self.config.data_name == 'MSD':
            vali_num = 8
            test_num = 10
        else:
            raise ValueError("data set not defined")
        tr_image_list = image_list[:len(image_list) - test_num]
        val_image_list = tr_image_list[len(tr_image_list) - vali_num:]
        tr_image_list = tr_image_list[:len(tr_image_list) - vali_num]
        train_size = self.config.default_seg_label_size
        tr_training = tr_image_list[:train_size]
        tr_training_un = tr_image_list[train_size:]
        train_transforms = Compose(
            [
                EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    ratios=[0.3, 0.7],
                    spatial_size=[self.config.patch_x, self.config.patch_y, self.config.patch_z],
                    num_classes=2,
                    num_samples=1,
                    allow_smaller=True
                ),
                ToTensord(keys=['image', 'label'])
            ]
        )
        w_train_transforms_1 = Compose(
            [
                EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    ratios=[0.3, 0.7],
                    spatial_size=[self.config.patch_x, self.config.patch_y, self.config.patch_z],
                    num_classes=2,
                    num_samples=1,
                    allow_smaller=True
                ),

            ]
        )
        w_train_transforms_2 = Compose(
            [
                RandFlipd(keys=['image', 'label'], prob=0.3, spatial_axis=0),
                RandRotated(keys=['image', 'label'], range_x=np.pi / 18,
                            range_y=np.pi / 18, prob=0.3),
                RandZoomd(keys=['image', 'label'], prob=0.3),
                RandAffined(
                    keys=['image', 'label'],
                    prob=0.3,
                    scale_range=(0.05, 0.05, 0.05)
                ),
            ]
        )
        s_train_transforms = Compose(
            [
                EnsureChannelFirstd(keys=['image', 'label', 'msk'], channel_dim='no_channel'),
                RandGaussianNoised(keys='image', prob=0.5),
                RandBiasFieldd(keys='image', prob=0.5),
                RandGibbsNoised(keys='image', prob=0.5),
                RandAdjustContrastd(keys='image', prob=0.5, gamma=(1.2, 2)),
                RandGaussianSmoothd(keys='image', prob=0.5),
                RandCoarseDropoutd(keys=['image', 'label', 'msk'], fill_value=0, holes=10, max_holes=30,
                                   spatial_size=20,
                                   prob=1),
            ]
        )

        db_train = XiTS(tr_training, base_dir=self.config.data_path,
                        transform=train_transforms, input_size=(self.config.patch_x, self.config.patch_y,
                                                                self.config.patch_z))
        train_dataloader = DataLoader(db_train, shuffle=True, collate_fn=self.merge_batch,
                                      batch_size=self.config.train_seg_batch_size, drop_last=True,
                                      num_workers=self.config.workers, pin_memory=True)
        val_dataset = XiTS(val_image_list, base_dir=self.config.data_path,
                           transform=None)
        if len(tr_training_un) > 0:
            train_dataset_un = XiWSTS(tr_training_un, base_dir=self.config.data_path,
                                      w_transform=[w_train_transforms_1, w_train_transforms_2],
                                      s_transform=s_train_transforms,
                                      input_size=(self.config.patch_x, self.config.patch_y,
                                                  self.config.patch_z))
            train_dataloader_un = DataLoader(train_dataset_un, batch_size=self.config.train_seg_batch_size,
                                             collate_fn=self.merge_my_batch, pin_memory=True, drop_last=True,
                                             shuffle=True, num_workers=self.config.workers)
        else:
            train_dataloader_un = None

        return train_dataloader, train_dataloader_un, val_dataset.data

    def merge_my_batch(self, batch):
        image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
        image_patch_s = [torch.unsqueeze(inst["image_patch_s"], dim=0) for inst in batch]
        image_segment = [torch.unsqueeze(inst["image_segment"], dim=0) for inst in batch]
        image_segment_s = [torch.unsqueeze(inst["image_segment_s"], dim=0) for inst in batch]
        image_msk = [torch.unsqueeze(inst["image_msk"], dim=0) for inst in batch]
        image_patch = torch.cat(image_patch, dim=0)
        image_patch_s = torch.cat(image_patch_s, dim=0)
        image_segment = torch.cat(image_segment, dim=0)
        image_segment_s = torch.cat(image_segment_s, dim=0)
        image_msk = torch.cat(image_msk, dim=0)
        return {"image_patch": image_patch,
                "image_patch_s": image_patch_s,
                "image_segment": image_segment,
                "image_segment_s": image_segment_s,
                "image_msk": image_msk
                }

    def read_my_data(self, dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                image_s = batch['image_patch_s']
                segment = batch['image_segment']
                segment_s = batch['image_segment_s']
                image_msk = batch['image_msk']
                image_s = image_s.to(self.equipment).float()
                image = image.to(self.equipment).float()
                segment = segment.to(self.equipment).long()
                segment_s = segment_s.to(self.equipment).long()
                image_msk = image_msk.to(self.equipment).long()
                yield {
                    "image_patch": image,
                    "image_patch_s": image_s,
                    "image_segment": segment,
                    "image_segment_s": segment_s,
                    "image_msk": image_msk
                }
