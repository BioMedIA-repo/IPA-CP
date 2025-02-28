# -*- coding: utf-8 -*-
# @Time    : 21/6/9 17:01
# @Software: PyCharm
# @Desc    : FistulaDataLoader.py
from commons.utils import *
from monai.data import Dataset
import torch
from copy import deepcopy
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandGaussianNoised,
    RandBiasFieldd,
    RandGibbsNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandAffined,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
    RandSpatialCropd
)

TMP_DIR = "./tmp"
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class FistulaBaseDataSet(Dataset):
    def __init__(self, data, transform=None, root_dir=None, text_only=False, input_size=None):
        fistula_df = data
        self.root_dir = root_dir
        self.transform = transform
        self.text_only = text_only
        self.input_size = input_size
        img_list = sorted(glob(join(root_dir, '*.npy')), reverse=False)
        seq_list = {}
        img_names = set()
        for fistula in fistula_df:
            # print(row['name'], row['训练组a验证组b'], row['Fistula'],row['序号'])
            seq_list[fistula['序号']] = {'name': fistula['name'],
                                         'Fistula': fistula['Fistula'],
                                         'num': fistula['num'],
                                         'cat': fistula['cat']
                                         }
        img_final_list = []
        for idx in range(len(img_list)):
            file_name = basename(img_list[idx])[:-4]
            split_dot = file_name.split('.')
            split_index = file_name.split(' ')
            if len(split_dot) == 2:
                # 1 + '.' + '2'
                set_idx = split_dot[0] + '.' + split_dot[1][0:1]
            elif len(split_index) == 2:
                # 1
                set_idx = split_index[0]
            else:
                continue
            if set_idx in seq_list:
                img_final_list.append({'seg': img_list[idx],
                                       'img': img_list[idx],
                                       'path': img_list[idx],
                                       'name': seq_list[set_idx]['name'],
                                       'Fistula': seq_list[set_idx]['Fistula'],
                                       'num': seq_list[set_idx]['num'],
                                       'cat': seq_list[set_idx]['cat'],
                                       '序号': set_idx})
                img_names.add(seq_list[set_idx]['name'])
        # self.img_final_list = img_final_list[:6]
        self.data = img_final_list
        print('Load images: %d' % (len(self.data)))
        print('Names:', sorted(list(img_names)))


class FistulaDataSet(FistulaBaseDataSet):
    def __init__(self, data, transform, root_dir, text_only, input_size):
        super(FistulaDataSet, self).__init__(data, transform, root_dir, text_only, input_size)

    def __getitem__(self, index):
        # while True:
        rand_data = self.data[index]
        file_name = basename(rand_data['path'])[:-4]
        scans = np.load(rand_data['path'])
        img = scans[0]
        fistula = scans[1]
        self.data[index]['img'] = img
        self.data[index]['seg'] = fistula
        patient = self.transform(self.data[index])
        seg_t = patient['seg']
        img_t = patient['img']
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(patient['seg'].numpy())
        deps = self.input_size[0]
        rows = self.input_size[1]
        cols = self.input_size[2]
        a = (maxx + minx) // 2
        b = (maxy + miny) // 2
        c = (maxz + minz) // 2
        a = a if a - deps // 2 >= 0 else deps // 2
        a = a if a + deps // 2 < seg_t.size(0) else seg_t.size(0) - deps // 2 - 1
        b = b if b - rows // 2 >= 0 else rows // 2
        b = b if b + rows // 2 < seg_t.size(1) else seg_t.size(1) - rows // 2 - 1
        c = c if c - cols // 2 >= 0 else cols // 2
        c = c if c + cols // 2 < seg_t.size(2) else seg_t.size(2) - cols // 2 - 1
        patient['img'] = img_t[max(0, a - deps // 2):min(seg_t.size(0), a + deps // 2),
                         max(0, b - rows // 2):min(seg_t.size(1), b + rows // 2), :]
        patient['seg'] = seg_t[max(0, a - deps // 2):min(seg_t.size(0), a + deps // 2),
                         max(0, b - rows // 2):min(seg_t.size(1), b + rows // 2), :]

        patient = EnsureChannelFirstd(keys=['img', 'seg', 'num', 'cat'], channel_dim='no_channel')(patient)

        resized = Resized(keys=['img', 'seg'], spatial_size=self.input_size)
        patient = resized(patient)
        patient['seg'] = torch.where(patient['seg'] > 0.1, 1, 0).int()
        return {
            "image_patch": patient['img'],
            "image_segment": patient['seg'],
        }


class FistulaWSDataSet(Dataset):
    def __init__(self, data, w_transform=None, s_transform=None, root_dir=None, input_size=None):
        fistula_df = data
        self.root_dir = root_dir
        self.w_transform = w_transform
        self.s_transform = s_transform
        self.input_size = input_size
        img_list = sorted(glob(join(root_dir, '*.npy')), reverse=False)
        seq_list = {}
        img_names = set()
        for fistula in fistula_df:
            # print(row['name'], row['训练组a验证组b'], row['Fistula'],row['序号'])
            seq_list[fistula['序号']] = {'name': fistula['name'],
                                         'Fistula': fistula['Fistula'],
                                         'num': fistula['num'],
                                         'cat': fistula['cat']
                                         }
        img_final_list = []
        for idx in range(len(img_list)):
            file_name = basename(img_list[idx])[:-4]
            split_dot = file_name.split('.')
            split_index = file_name.split(' ')
            if len(split_dot) == 2:
                # 1 + '.' + '2'
                set_idx = split_dot[0] + '.' + split_dot[1][0:1]
            elif len(split_index) == 2:
                # 1
                set_idx = split_index[0]
            else:
                continue
            if set_idx in seq_list:
                img_final_list.append({'seg': img_list[idx],
                                       'img': img_list[idx],
                                       'path': img_list[idx],
                                       'name': seq_list[set_idx]['name'],
                                       'Fistula': seq_list[set_idx]['Fistula'],
                                       'num': seq_list[set_idx]['num'],
                                       'cat': seq_list[set_idx]['cat'],
                                       '序号': set_idx})
                img_names.add(seq_list[set_idx]['name'])
        # self.img_final_list = img_final_list[:6]
        self.data = img_final_list
        print('Load images: %d' % (len(self.data)))
        print('Names:', sorted(list(img_names)))

    def __getitem__(self, index):
        # while True:
        rand_data = self.data[index]
        file_name = basename(rand_data['path'])[:-4]
        scans = np.load(rand_data['path'])
        img = scans[0]
        fistula = scans[1]
        self.data[index]['img'] = img
        self.data[index]['seg'] = fistula
        ori = deepcopy(self.data[index])
        w_patient = self.w_transform(self.data[index])
        s_patient = deepcopy(w_patient)
        mask = torch.ones(s_patient['seg'].shape)
        s_patient['msk'] = mask
        s_patient = self.s_transform(s_patient)
        s_patient['msk'] = s_patient['msk'].squeeze()
        s_patient['img'] = s_patient['img'].squeeze()
        s_patient['seg'] = s_patient['seg'].squeeze()

        minx, maxx, miny, maxy, minz, maxz = min_max_voi(w_patient['seg'].numpy())
        deps = self.input_size[0]
        rows = self.input_size[1]
        cols = self.input_size[2]
        a = (maxx + minx) // 2
        b = (maxy + miny) // 2
        c = (maxz + minz) // 2
        a = a if a - deps // 2 >= 0 else deps // 2
        a = a if a + deps // 2 < w_patient['seg'].size(0) else w_patient['seg'].size(0) - deps // 2 - 1
        b = b if b - rows // 2 >= 0 else rows // 2
        b = b if b + rows // 2 < w_patient['seg'].size(1) else w_patient['seg'].size(1) - rows // 2 - 1
        # c = c if c - cols // 2 >= 0 else cols // 2
        # c = c if c + cols // 2 < w_patient['seg'].size(2) else w_patient['seg'].size(2) - cols // 2 - 1
        ad, bd = w_patient['seg'].size(0), w_patient['seg'].size(1)

        ori['img'] = ori['img'][max(0, a - deps // 2):min(ad, a + deps // 2),
                     max(0, b - rows // 2):min(bd, b + rows // 2), :]
        w_patient['img'] = w_patient['img'][max(0, a - deps // 2):min(ad, a + deps // 2),
                           max(0, b - rows // 2):min(bd, b + rows // 2), :]
        w_patient['seg'] = w_patient['seg'][max(0, a - deps // 2):min(ad, a + deps // 2),
                           max(0, b - rows // 2):min(bd, b + rows // 2), :]
        s_patient['img'] = s_patient['img'][max(0, a - deps // 2):min(ad, a + deps // 2),
                           max(0, b - rows // 2):min(bd, b + rows // 2), :]
        s_patient['seg'] = s_patient['seg'][max(0, a - deps // 2):min(ad, a + deps // 2),
                           max(0, b - rows // 2):min(bd, b + rows // 2), :]
        s_patient['msk'] = s_patient['msk'][max(0, a - deps // 2):min(ad, a + deps // 2),
                           max(0, b - rows // 2):min(bd, b + rows // 2), :]
        ori = EnsureChannelFirstd(keys=['img', 'seg', 'num', 'cat'], channel_dim='no_channel')(ori)
        w_patient = EnsureChannelFirstd(keys=['img', 'seg', 'num', 'cat'], channel_dim='no_channel')(w_patient)
        s_patient = EnsureChannelFirstd(keys=['img', 'seg', 'msk', 'num', 'cat'], channel_dim='no_channel')(s_patient)
        ori = ToTensord(keys=['img', 'seg', 'num', 'cat'])(ori)
        w_patient = ToTensord(keys=['img', 'seg', 'num', 'cat'])(w_patient)
        s_patient = ToTensord(keys=['img', 'seg', 'msk', 'num', 'cat'])(s_patient)

        resized = Resized(keys=['img', 'seg'], spatial_size=self.input_size)
        w_patient = resized(w_patient)
        ori = resized(ori)
        w_patient['seg'] = torch.round(w_patient['seg']).int()
        resized_s = Resized(keys=['img', 'seg', 'msk'], spatial_size=self.input_size)
        s_patient = resized_s(s_patient)
        s_patient['seg'] = torch.round(s_patient['seg']).int()
        s_patient['msk'] = torch.round(s_patient['msk']).int()

        return {
            "image_ori":ori['img'],
            "image_patch": w_patient['img'],
            "image_patch_s": s_patient['img'],
            "image_segment": w_patient['seg'],
            "image_segment_s": s_patient['seg'],
            "image_msk": s_patient['msk'],
        }


class FistulaTestDataSet(FistulaBaseDataSet):
    def __init__(self, data, transform, root_dir, text_only, input_size):
        super(FistulaTestDataSet, self).__init__(data, transform, root_dir, text_only, input_size)


if __name__ == '__main__':
    from monai.data import DataLoader
    from fistula_data.preprocess.prepare_fistula import read_csv
    from monai.utils import first

    csv_data_path = "../../medical_data/esophageal/esophageal_fistula.csv"
    data_path = "../../medical_data/esophageal/fistula-Seg_Processed_300"


    def read_data(dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                image_cls = batch['image_cls']
                label = batch['image_label']
                segment = batch['image_segment']
                name = batch['image_name']
                num = batch['image_num']
                cat = batch['image_cat']
                yield {
                    "image_cls": image_cls,
                    "image_patch": image,
                    "image_label": label,
                    "image_segment": segment,
                    "image_name": name,
                    "image_num": num,
                    "image_cat": cat
                }


    def merge_batch(batch):
        image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
        image_cls = [torch.unsqueeze(inst["image_cls"], dim=0) for inst in batch]
        image_label = [inst["image_label"] for inst in batch]
        image_segment = [torch.unsqueeze(inst["image_segment"], dim=0) for inst in batch]
        image_num = [torch.unsqueeze(inst["image_num"], dim=0) for inst in batch]
        image_cat = [torch.unsqueeze(inst["image_cat"], dim=0) for inst in batch]
        image_patch = torch.cat(image_patch, dim=0)
        image_cls = torch.cat(image_cls, dim=0)
        image_label = torch.tensor(image_label)
        image_segment = torch.cat(image_segment, dim=0)
        image_num = torch.cat(image_num, dim=0)
        image_cat = torch.cat(image_cat, dim=0)
        image_name = [inst["image_name"] for inst in batch]
        return {"image_patch": image_patch,
                "image_cls": image_cls,
                "image_label": image_label,
                "image_segment": image_segment,
                "image_name": image_name,
                "image_num": image_num,
                "image_cat": image_cat
                }


    # Rand3DElasticd,
    # Rand3DElasticd(
    #     keys=["image", "label"],
    #     mode=("bilinear", "nearest"),
    #     prob=1.0,
    #     sigma_range=(5, 8),
    #     magnitude_range=(100, 200),
    #     spatial_size=(300, 300, 10),
    #     translate_range=(50, 50, 2),
    #     rotate_range=(np.pi / 36, np.pi / 36, np.pi),
    #     scale_range=(0.15, 0.15, 0.15),
    #     padding_mode="border",
    # )

    train_transforms = Compose(
        [
            RandGaussianNoised(keys='img', prob=0.3),
            RandBiasFieldd(keys='img', prob=0.3),
            RandGibbsNoised(keys='img', prob=0.3),
            RandAdjustContrastd(keys='img', prob=0.3, gamma=(1.2, 2)),
            RandGaussianSmoothd(keys='img', prob=0.3),
            # RandFlipd(keys=['img', 'seg'], prob=1, spatial_axis=0),
            # RandRotated(keys=['img', 'seg'], range_x=np.pi / 18,
            #             range_y=np.pi / 18, prob=1),
            # RandZoomd(keys=['img', 'seg'], prob=1),
            # RandAffined(
            #     keys=['img', 'seg'],
            #     prob=1,
            #     scale_range=(0.05, 0.05, 0.05)
            # ),
            ToTensord(keys=['img', 'seg', 'num', 'cat'])
        ]
    )
    fistula_training, fistula_test = read_csv(csv_data_path)
    test_dataset = FistulaDataSet(data=fistula_training, transform=train_transforms, root_dir=data_path,
                                  text_only=False, input_size=(224, 224, 80))
    test_loader_mtl = DataLoader(test_dataset, batch_size=2, collate_fn=merge_batch,
                                 shuffle=False)
    # test_patient = first(test_loader_mtl)
    # visual_batch(test_patient['img'], './tmp', "1_images", channel=1, nrow=8)
    # visual_batch(test_patient['seg'], './tmp', "1_gt" + '-ca', channel=1, nrow=8)
    for i, batch in enumerate(test_loader_mtl):
        images = batch['image_patch']
        image_cls = batch['image_cls']
        labels = batch['image_label']
        segments = batch['image_segment']
        names = batch['image_name']
        image_num = batch['image_num']
        image_cat = batch['image_cat']

        visual_batch(images, './tmp', "b_%d_images" % (i), channel=1, nrow=16)
        visual_batch(segments, './tmp', "b_%d_labels" % (i), channel=1,
                     nrow=16)
