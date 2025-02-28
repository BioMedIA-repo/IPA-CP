import torch
from driver.base_train_helper import BaseTrainHelper
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from fistula_data.FistulaDataLoader import FistulaDataSet, FistulaTestDataSet, FistulaWSDataSet
import numpy as np
import SimpleITK as sitk
import cv2
from os.path import exists, join
from einops import rearrange, reduce, repeat
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
    RandRotated,
    RandZoomd,
    RandAffined,
    RandCoarseDropoutd,
    RandSpatialCropd,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
)
from commons.utils import *
from torch.cuda import empty_cache
from torch.nn.utils import clip_grad_norm_
from commons.evaluation import compute_all_metric_for_single_seg
from skimage.transform import resize
import matplotlib


plt.rcParams.update({'figure.max_open_warning': 20})


class FistulaHelper(BaseTrainHelper):
    def __init__(self, criterions, config):
        super(FistulaHelper, self).__init__(criterions, config)

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
        train_transforms = Compose(
            [
                # RandGaussianNoised(keys='img', prob=0.3),
                # RandBiasFieldd(keys='img', prob=0.3),
                # RandGibbsNoised(keys='img', prob=0.3),
                # RandAdjustContrastd(keys='img', prob=0.3, gamma=(1.2, 2)),
                # RandGaussianSmoothd(keys='img', prob=0.3),
                # RandFlipd(keys=['img', 'seg'], prob=0.3, spatial_axis=0),
                # RandRotated(keys=['img', 'seg'], range_x=np.pi / 18,
                #             range_y=np.pi / 18, prob=0.3),
                # RandZoomd(keys=['img', 'seg'], prob=0.3),
                # RandAffined(
                #     keys=['img', 'seg'],
                #     prob=0.3,
                #     scale_range=(0.05, 0.05, 0.05)
                # ),
                ToTensord(keys=['img', 'seg', 'num', 'cat']),
            ]
        )

        from fistula_data.preprocess.prepare_fistula import read_csv
        fistula_training, _ = read_csv(self.config.csv_data_path)
        np.random.seed(seed)
        np.random.shuffle(fistula_training)
        # 360 samples
        fistula_val = fistula_training[:30]
        fistula_final_tr = fistula_training[30:]
        train_size = self.config.default_seg_label_size
        fistula_training = fistula_final_tr[:train_size]
        fistula_training_un = fistula_final_tr[train_size:]
        train_dataset = FistulaDataSet(data=fistula_training, transform=train_transforms,
                                       root_dir=self.config.data_path,
                                       text_only=False,
                                       input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z))
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.train_seg_batch_size,
                                      collate_fn=self.merge_batch, pin_memory=True, drop_last=True,
                                      shuffle=True, num_workers=self.config.workers)
        val_dataset = FistulaTestDataSet(data=fistula_val, transform=None,
                                         root_dir=self.config.data_path,
                                         text_only=False,
                                         input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z))
        if len(fistula_training_un) > 0:
            train_dataset_un = FistulaDataSet(data=fistula_training_un, transform=train_transforms,
                                              root_dir=self.config.data_path,
                                              text_only=False,
                                              input_size=(
                                                  self.config.patch_x, self.config.patch_y, self.config.patch_z))
            train_dataloader_un = DataLoader(train_dataset_un, batch_size=self.config.train_seg_batch_size,
                                             collate_fn=self.merge_batch, pin_memory=True, drop_last=True,
                                             shuffle=True, num_workers=self.config.workers)
        else:
            train_dataloader_un = None
        return train_dataloader, train_dataloader_un, val_dataset.data

    def get_test_data_loader(self, text_only=False):
        from fistula_data.preprocess.prepare_fistula import read_csv
        _, fistula_test = read_csv(self.config.csv_data_path)
        test_dataset = FistulaTestDataSet(data=fistula_test, transform=None,
                                          root_dir=self.config.data_path,
                                          text_only=text_only,
                                          input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z))
        return test_dataset.data

    def max_norm_cam(self, cam_g, e=1e-5):
        saliency_map = np.maximum(0, cam_g)
        saliency_map_min, saliency_map_max = np.min(saliency_map), np.max(saliency_map)
        cam_out = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min + e)
        cam_out = np.maximum(0, cam_out)
        return cam_out


    def predict_whole(self, pmodel, test_data, vis=False, isTest=False):
        img_cols = self.config.patch_x
        img_rows = self.config.patch_y
        img_deps = self.config.patch_z
        fistula_segmentation_metrics = {}
        fistula_segmentation_scores = {}
        total_metric = np.zeros((self.config.classes - 1, 2))
        if vis:
            output_dir = join(self.config.tmp_dir, 'infer')
            mkdir_if_not_exist([output_dir])
        for file_path_dict in test_data:
            # ----------------TEXT-----------------
            # print(file_path)
            file_name = basename(file_path_dict['path'])
            # print(file_name)
            scans = np.load(file_path_dict['path'])
            img = scans[0]
            current_test = img.copy()
            fistula_seg_img = scans[1]
            nrow = 8
            minx, maxx, miny, maxy, minz, maxz = min_max_voi(fistula_seg_img, superior=20, inferior=20)
            midx = (maxx + minx) // 2
            midy = (maxy + miny) // 2
            midz = (maxz + minz) // 2
            midx = midx if midx - img_cols // 2 >= 0 else img_cols // 2
            midx = midx if midx + img_cols // 2 < img.shape[0] else img.shape[0] - img_cols // 2 - 1
            midy = midy if midy - img_rows // 2 >= 0 else img_rows // 2
            midy = midy if midy + img_rows // 2 < img.shape[1] else img.shape[1] - img_rows // 2 - 1
            midz = midz if midz - img_deps // 2 >= 0 else img_deps // 2
            midz = midz if midz + img_deps // 2 < img.shape[2] else img.shape[2] - img_deps // 2 - 1
            img_crop = img[int(midx - img_cols / 2):int(midx + img_cols / 2),
                       int(midy - img_rows / 2):int(midy + img_rows / 2),
                       max(0, int(midz - img_deps / 2)): min(img.shape[2], int(midz + img_deps / 2))]
            seg_crop = fistula_seg_img[int(midx - img_cols / 2):int(midx + img_cols / 2),
                       int(midy - img_rows / 2):int(midy + img_rows / 2),
                       max(0, int(midz - img_deps / 2)): min(img.shape[2], int(midz + img_deps / 2))]
            cropp_fistula_img = resize(seg_crop, (img_cols, img_rows, img_deps), order=0, mode='edge',
                                       cval=0, clip=True, preserve_range=True, anti_aliasing=False)
            cropp_img = resize(img_crop, (img_cols, img_rows, img_deps), order=3, mode='constant',
                               cval=0, clip=True, preserve_range=True, anti_aliasing=False)
            box_test = torch.from_numpy(cropp_img)
            box_test = torch.unsqueeze(torch.unsqueeze(box_test, dim=0), dim=0)
            box_test = box_test.to(self.equipment).float()

            cropp_fistula_img_np = cropp_fistula_img.copy()
            cropp_fistula_img = torch.from_numpy(cropp_fistula_img)
            # cropp_fistula_img = rearrange(cropp_fistula_img, 'h w d -> () () h w d')
            cropp_fistula_img = torch.unsqueeze(torch.unsqueeze(cropp_fistula_img, dim=0), dim=0)
            cropp_fistula_img = cropp_fistula_img.to(self.equipment).float()
            if vis:
                visual_batch(box_test, output_dir,
                             '%s_images_%d_label_raw' % (
                                 file_name[:-4], file_path_dict['Fistula']),
                             channel=1,
                             nrow=nrow)
                # visual_batch(box_test, output_dir,
                #              '%s_images_%d_label_rawcls' % (
                #                  file_name[:-4], file_path_dict['Fistula']),
                #              channel=1,
                #              nrow=nrow)
                visual_batch(cropp_fistula_img, output_dir, '%s_labels_%d_label_gt' % (
                    file_name[:-4], file_path_dict['Fistula']),
                             channel=1, nrow=nrow)
            with torch.no_grad():
                # FOR MCF
                if isinstance(pmodel, list):
                    patch_test_mask = pmodel[0](box_test)
                    y2 = pmodel[1](box_test)
                    prob_maps = torch.softmax(patch_test_mask, dim=1)
                    prob_maps_r = torch.softmax(y2, dim=1)
                    prob_maps = (prob_maps + prob_maps_r) / 2
                elif isinstance(pmodel, tuple):
                    # FOR MCF
                    patch_test_mask = pmodel[0](box_test)
                    y2 = pmodel[1](box_test)
                    prob_maps = torch.softmax(patch_test_mask, dim=1)
                    prob_maps_r = torch.softmax(y2, dim=1)
                    prob_maps = (prob_maps + prob_maps_r) / 2
                else:
                    patch_test_mask = pmodel(box_test)
                    # FOR CCT
                    if self.config.model == 'asnet':
                        patch_test_mask = torch.sigmoid(patch_test_mask)
                        prob_maps = torch.cat([1 - patch_test_mask, patch_test_mask], dim=1)
                    else:
                        if isinstance(patch_test_mask, list):
                            patch_test_mask = patch_test_mask[0]
                            prob_maps = torch.softmax(patch_test_mask, dim=1)
                        elif isinstance(patch_test_mask, tuple):
                            patch_test_mask = patch_test_mask[0]
                            prob_maps = torch.softmax(patch_test_mask, dim=1)
                        else:
                            prob_maps = torch.softmax(patch_test_mask, dim=1)
            fine_pred = torch.argmax(prob_maps, dim=1).squeeze()
            fine_pred = fine_pred.cpu().detach().numpy()
            if isTest:
                scores = compute_all_metric_for_single_seg(cropp_fistula_img_np, fine_pred)
                for metric in scores:
                    if metric not in fistula_segmentation_scores:
                        fistula_segmentation_scores[metric] = []
                    fistula_segmentation_scores[metric].extend(scores[metric])
            else:
                for i in range(1, self.config.classes):
                    total_metric[i - 1, :] += self.cal_metric(cropp_fistula_img_np == i, fine_pred == i)
            if vis:
                visual_batch(prob_maps[:, 1, :, :, :].unsqueeze(1), output_dir, '%s_labels_%d_label_prob' % (
                    file_name[:-4], file_path_dict['Fistula']),
                             channel=1, nrow=nrow)
                visual_batch(torch.argmax(prob_maps, dim=1, keepdim=True), output_dir, '%s_labels_%d_label_pred' % (
                    file_name[:-4], file_path_dict['Fistula']),
                             channel=1, nrow=nrow)
                pred_itk = sitk.GetImageFromArray(fine_pred.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, output_dir +
                                "/{}_pred.nii.gz".format(file_name))

                img_itk = sitk.GetImageFromArray(cropp_img)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, output_dir +
                                "/{}_img.nii.gz".format(file_name))

                lab_itk = sitk.GetImageFromArray(cropp_fistula_img_np.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, output_dir +
                                "/{}_lab.nii.gz".format(file_name))
                # TODO close this for faster calculation
                # vis_raw = box_test.squeeze().detach().cpu().numpy()
                # vis_raw = rearrange(vis_raw, 'h w d -> d h w')
                # cropp_fistula_vis = rearrange(cropp_fistula_img_np, 'h w d -> d h w')
                # fine_pred_vis = rearrange(fine_pred, 'h w d -> d h w')
                # for i in range(len(vis_raw)):
                #     cropp_fistula_i = cropp_fistula_vis[i]
                #     fine_pred_i = fine_pred_vis[i]
                #     if len(np.unique(cropp_fistula_i)) > 1 or len(np.unique(fine_pred_i)) > 1:
                #         miccaiimshow(vis_raw[i], cropp_fistula_vis[i], fine_pred_vis[i],
                #                      fname=join(output_dir, '%s_labels_%d_label_differ_%02d.png' % (
                #                          file_name[:-4], file_path_dict['Fistula'], i)))
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
        else:
            seg_dice = total_metric[0, 0] / len(test_data)
            print("Valid: Dice {}, HD {}".format(total_metric[0, 0] / len(test_data),
                                                 total_metric[0, 1] / len(test_data)))
        if vis:
            zipDir(output_dir, output_dir + '.zip')
            shutil.rmtree(output_dir)
        return {
            'vali/seg_acc': seg_dice,
            'pred_acc': fistula_segmentation_metrics
        }

    def merge_my_batch(self, batch):
        image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
        ori_patch = [torch.unsqueeze(inst["image_ori"], dim=0) for inst in batch]
        image_patch_s = [torch.unsqueeze(inst["image_patch_s"], dim=0) for inst in batch]
        image_segment = [torch.unsqueeze(inst["image_segment"], dim=0) for inst in batch]
        image_segment_s = [torch.unsqueeze(inst["image_segment_s"], dim=0) for inst in batch]
        image_msk = [torch.unsqueeze(inst["image_msk"], dim=0) for inst in batch]
        image_patch = torch.cat(image_patch, dim=0)
        ori_patch = torch.cat(ori_patch, dim=0)
        image_patch_s = torch.cat(image_patch_s, dim=0)
        image_segment = torch.cat(image_segment, dim=0)
        image_segment_s = torch.cat(image_segment_s, dim=0)
        image_msk = torch.cat(image_msk, dim=0)
        return {"image_ori": ori_patch,
                "image_patch": image_patch,
                "image_patch_s": image_patch_s,
                "image_segment": image_segment,
                "image_segment_s": image_segment_s,
                "image_msk": image_msk
                }

    def read_my_data(self, dataloader):
        while True:
            for batch in dataloader:
                ori = batch['image_ori']
                image = batch['image_patch']
                image_s = batch['image_patch_s']
                segment = batch['image_segment']
                segment_s = batch['image_segment_s']
                image_msk = batch['image_msk']
                image_s = image_s.to(self.equipment).float()
                image = image.to(self.equipment).float()
                ori = ori.to(self.equipment).float()
                image_s = image_s.to(self.equipment).float()
                segment = segment.to(self.equipment).long()
                segment_s = segment_s.to(self.equipment).long()
                image_msk = image_msk.to(self.equipment).long()
                yield {
                    "image_ori": ori,
                    "image_patch": image,
                    "image_patch_s": image_s,
                    "image_segment": segment,
                    "image_segment_s": segment_s,
                    "image_msk": image_msk
                }

    def get_ws_train_loader(self, seed=1337):
        train_transforms = Compose(
            [
                # RandFlipd(keys=['img', 'seg'], prob=0.3, spatial_axis=0),
                # RandRotated(keys=['img', 'seg'], range_x=np.pi / 18,
                #             range_y=np.pi / 18, prob=0.3),
                # RandZoomd(keys=['img', 'seg'], prob=0.3),
                # RandAffined(
                #     keys=['img', 'seg'],
                #     prob=0.3,
                #     scale_range=(0.05, 0.05, 0.05)
                # ),
                ToTensord(keys=['img', 'seg', 'num', 'cat']),
            ]
        )

        w_train_transforms = Compose(
            [
                RandFlipd(keys=['img', 'seg'], prob=0.3, spatial_axis=0),
                RandRotated(keys=['img', 'seg'], range_x=np.pi / 18,
                            range_y=np.pi / 18, prob=0.3),
                RandZoomd(keys=['img', 'seg'], prob=0.3),
                RandAffined(
                    keys=['img', 'seg'],
                    prob=0.3,
                    scale_range=(0.05, 0.05, 0.05)
                ),
            ]
        )
        s_train_transforms = Compose(
            [
                EnsureChannelFirstd(keys=['img', 'seg', 'msk', 'num', 'cat'], channel_dim='no_channel'),
                RandGaussianNoised(keys='img', prob=0.5),
                RandBiasFieldd(keys='img', prob=0.5),
                RandGibbsNoised(keys='img', prob=0.5),
                RandAdjustContrastd(keys='img', prob=0.5, gamma=(1.2, 2)),
                RandGaussianSmoothd(keys='img', prob=0.5),
                RandCoarseDropoutd(keys=['img', 'seg', 'msk'], fill_value=0, holes=5, max_holes=40,
                                   spatial_size=10, max_spatial_size=40,
                                   prob=1),
            ]
        )
        from fistula_data.preprocess.prepare_fistula import read_csv
        fistula_training, _ = read_csv(self.config.csv_data_path)
        np.random.seed(seed)
        np.random.shuffle(fistula_training)
        # 360 samples
        fistula_val = fistula_training[:30]
        fistula_final_tr = fistula_training[30:]
        train_size = self.config.default_seg_label_size
        fistula_training = fistula_final_tr[:train_size]
        fistula_training_un = fistula_final_tr[train_size:]
        train_dataset = FistulaDataSet(data=fistula_training, transform=train_transforms,
                                       root_dir=self.config.data_path,
                                       text_only=False,
                                       input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z))
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.train_seg_batch_size,
                                      collate_fn=self.merge_batch, pin_memory=True, drop_last=True,
                                      shuffle=True, num_workers=self.config.workers)
        val_dataset = FistulaTestDataSet(data=fistula_val, transform=None,
                                         root_dir=self.config.data_path,
                                         text_only=False,
                                         input_size=(self.config.patch_x, self.config.patch_y, self.config.patch_z))
        if len(fistula_training_un) > 0:
            train_dataset_un = FistulaWSDataSet(data=fistula_training_un, w_transform=w_train_transforms,
                                                s_transform=s_train_transforms,
                                                root_dir=self.config.data_path,
                                                input_size=(
                                                    self.config.patch_x, self.config.patch_y, self.config.patch_z))
            train_dataloader_un = DataLoader(train_dataset_un, batch_size=self.config.train_seg_batch_size,
                                             collate_fn=self.merge_my_batch, pin_memory=True, drop_last=True,
                                             shuffle=True, num_workers=self.config.workers)
        else:
            train_dataloader_un = None
        return train_dataloader, train_dataloader_un, val_dataset.data
