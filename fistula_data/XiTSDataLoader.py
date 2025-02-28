# -*- coding: utf-8 -*-

from fistula_data.utils import *
from monai.transforms import (
    Resized,
    EnsureChannelFirstd,
    ToTensord,
)
import time
from einops import rearrange, reduce, repeat
from copy import deepcopy


class XiTS(Dataset):
    """ LiTS2017 and KiTS Dataset """

    def __init__(self, image_list, base_dir=None, transform=None, input_size=(96, 96, 96)):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = image_list
        self.data = image_list
        self.input_size = input_size
        self.resized = Resized(keys=['image', 'label'], spatial_size=self.input_size)
        img_names = set()
        for name in image_list:
            img_names.add(basename(name))
        print("Total {} samples".format(len(self.image_list)))
        print('Names:', sorted(list(img_names)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img = np.load(image_name)
        img_ori = img[..., 0]
        seg = img[..., 1]
        sample = {'image': img_ori, 'label': seg.astype(np.uint8)}
        if self.transform:
            # print("Time transform")
            # now_time = time.time()
            # print("Time end %.4f" % (time.time() - now_time))
            # sample = self.transform(sample)
            sample = self.transform(sample)[0]
            sample = self.resized(sample)
            sample['label'] = torch.round(sample['label'])
        return {
            "image_patch": sample['image'],
            "image_segment": sample['label'],
        }


class XiWSTS(Dataset):
    """ LiTS2017 and KiTS WS Dataset """

    def __init__(self, image_list, base_dir=None, w_transform=None, s_transform=None, input_size=(96, 96, 96)):
        self._base_dir = base_dir
        self.w_transform = w_transform
        self.s_transform = s_transform
        self.image_list = image_list
        self.data = image_list
        self.input_size = input_size
        self.resized = Resized(keys=['image', 'label'], spatial_size=self.input_size)
        self.resized_s = Resized(keys=['image', 'label','msk'], spatial_size=self.input_size)
        self.addC = EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel')
        img_names = set()
        for name in image_list:
            img_names.add(basename(name))
        print("Total {} samples".format(len(self.image_list)))
        print('Names:', sorted(list(img_names)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img = np.load(image_name)
        img_ori = img[..., 0]
        seg = img[..., 1]
        sample = {'image': img_ori, 'label': seg.astype(np.uint8)}
        if self.w_transform and self.s_transform:
            w_patient = self.w_transform[0](sample)[0]
            w_patient['image'] = rearrange(w_patient['image'], '() w h d -> w h d')
            w_patient['label'] = rearrange(w_patient['label'], '() w h d -> w h d')
            w_patient = self.w_transform[1](w_patient)
            w_patient = self.addC(w_patient)
            w_patient = self.resized(w_patient)
            w_patient['label'] = torch.round(w_patient['label'])
            w_patient_tensor = ToTensord(keys=['image', 'label'])(w_patient)
            w_patient['image'] = rearrange(w_patient['image'], '() w h d -> w h d')
            w_patient['label'] = rearrange(w_patient['label'], '() w h d -> w h d')
            s_patient = deepcopy(w_patient)
            mask = torch.ones(w_patient['label'].shape)
            s_patient['msk'] = mask
            s_patient = self.s_transform(s_patient)
            s_patient = self.resized_s(s_patient)
            s_patient['label'] = torch.round(s_patient['label'])
            s_patient['msk'] = torch.round(s_patient['msk'])
            s_patient = ToTensord(keys=['image', 'label','msk'])(s_patient)
        return {
            "image_patch": w_patient_tensor['image'],
            "image_patch_s": s_patient['image'],
            "image_segment": w_patient_tensor['label'],
            "image_segment_s": s_patient['label'],
            "image_msk": s_patient['msk'],
        }


if __name__ == '__main__':
    from monai.data import DataLoader
    from torchvision import transforms

    from monai.transforms import (
        Compose,
        ToTensord,
        EnsureChannelFirstd,
        RandCropByLabelClassesd
    )

    data_path = "../../medical_data/LiTS/processed_222"
    input_size = (224, 224, 32)


    def merge_batch(batch):
        image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
        image_segment = [torch.unsqueeze(inst["image_segment"], dim=0) for inst in batch]
        image_patch = torch.cat(image_patch, dim=0)
        image_segment = torch.cat(image_segment, dim=0)
        return {"image_patch": image_patch,
                "image_segment": image_segment,
                }


    def read_data(dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                segment = batch['image_segment']
                yield {
                    "image_patch": image,
                    "image_segment": segment,
                }


    train_transforms = Compose(
        [
            # RandGaussianNoised(keys='image', prob=0.3),
            # RandBiasFieldd(keys='image', prob=0.3),
            # RandGibbsNoised(keys='image', prob=0.3),
            # RandAdjustContrastd(keys='image', prob=0.3, gamma=(1.2, 2)),
            # RandGaussianSmoothd(keys='image', prob=0.3),
            # RandFlipd(keys=['img', 'seg'], prob=1, spatial_axis=0),
            # RandRotated(keys=['img', 'seg'], range_x=np.pi / 18,
            #             range_y=np.pi / 18, prob=1),
            # RandZoomd(keys=['img', 'seg'], prob=1),
            # RandAffined(
            #     keys=['img', 'seg'],
            #     prob=1,
            #     scale_range=(0.05, 0.05, 0.05)
            # ),
            EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
            # RandCropByLabelClassesd(
            #     keys=["image", "label"],
            #     label_key="label",
            #     ratios=[0.3, 0.7],
            #     spatial_size=input_size,
            #     num_classes=2,
            #     num_samples=1,
            #     allow_smaller=True
            # ),
            ToTensord(keys=['image', 'label'])
        ]
    )
    image_list = sorted(glob(join(data_path, '*.npy')), reverse=False)

    # test_dataset = XiTS(image_list, base_dir=data_path,
    #                     transform=None)
    test_dataset = XiTS(image_list, base_dir=data_path,
                        transform=train_transforms, input_size=input_size)
    test_loader_mtl = DataLoader(test_dataset, batch_size=1, collate_fn=merge_batch,
                                 shuffle=False)
    for i, batch in enumerate(test_loader_mtl):
        images = batch['image_patch']
        segments = batch['image_segment']
        print(images.size())
        # print(segments.size())
        # visual_batch(images, './tmp', "b_%d_images" % (i), channel=1, nrow=16)
        # visual_batch(segments, './tmp', "b_%d_labels" % (i), channel=1,
        #              nrow=16)
