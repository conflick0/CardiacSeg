# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f'load json from {file_path}')
        return data


def load_data_dicts(json_pth, mode):
    data = load_json(json_pth)
    if mode == 'training':
        return data['training']
    elif mode == 'validation':
        return data['validation']
    elif mode == 'test':
        return data['test']
    else:
        raise KeyError()


def get_loader(args):
    data_json_pth1 = "/content/drive/MyDrive/CardiacSeg/dataset/SegTHOR/dataset.json"
    data_json_pth2 = "/content/drive/MyDrive/CardiacSeg/dataset/MM_WHS_2017/dataset.json"

    num_workers = 2

    datalist1 = load_data_dicts(data_json_pth1, 'training')
    datalist2 = load_data_dicts(data_json_pth2, 'training')

    vallist1 = load_data_dicts(data_json_pth1, 'validation')
    vallist2 = load_data_dicts(data_json_pth2, 'validation')

    print("Dataset 1 SegTHOR: number of tr data: {}".format(len(datalist1)))
    print("Dataset 2 MMWHS: number of tr data: {}".format(len(datalist2)))

    print("Dataset 1 SegTHOR: number of val data: {}".format(len(vallist1)))
    print("Dataset 2 MMWHS: number of val data: {}".format(len(vallist2)))

    datalist = datalist1 + datalist2
    val_files = vallist1 + vallist2
    
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=datalist, transform=train_transforms)


    train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader
