import os
from posix import times_result
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    LabelFilterd,
    MapLabelValued
)
from monai.data import (
    DataLoader,
    CacheDataset
)


def get_ds_pths(data_dir):
    img_pths = []
    lbl_pths = []
    for patient_dir in sorted(os.listdir(data_dir)):
        img_pths.append(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz'))
        lbl_pths.append(os.path.join(data_dir, patient_dir, f'GT.nii.gz'))
    return img_pths, lbl_pths


def get_data_dicts(data_dir):
    img_pths, lbl_pths = get_ds_pths(data_dir)
    data_dicts = [
        {"image": img_pth, "label": lbl_pth}
        for img_pth, lbl_pth in zip(img_pths, lbl_pths)
    ]
    return data_dicts


def get_train_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            LabelFilterd(keys=["label"], applied_labels=[2]),
            MapLabelValued(keys=["label"], orig_labels=[0, 2], target_labels=[0, 1]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=['image', 'label'], source_key='label', margin=40),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transform():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            LabelFilterd(keys=["label"], applied_labels=[2]),
            MapLabelValued(keys=["label"], orig_labels=[0, 2], target_labels=[0, 1]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=['image', 'label'], source_key='label', margin=40),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_data_loader(files, transform, shuffle, args):
    ds = CacheDataset(
        data=files,
        transform=transform,
        num_workers=args.workers
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )
    return loader

def get_train_val_test_files(data_dir):
    data_dicts = get_data_dicts(data_dir)
    train_files, val_files, test_files = data_dicts[:26], data_dicts[26:32], data_dicts[32:]
    return train_files, val_files, test_files

def get_loader(args):
    train_files, val_files, test_files = get_train_val_test_files(args.data_dir)
    
    if args.test_mode:
        print("num test files:", len(test_files))
        val_transform = get_val_transform()
        test_loader = get_data_loader(
            files=test_files,
            transform=val_transform,
            shuffle=False,
            args=args
        )
        return [test_loader]
    else:
        print("num train files:", len(train_files))
        print("num val files:", len(val_files))

        train_transform = get_train_transform(args)
        val_transform = get_val_transform()

        train_loader = get_data_loader(
            files=train_files,
            transform=train_transform,
            shuffle=True,
            args=args
        )
        val_loader = get_data_loader(
            files=val_files,
            transform=val_transform,
            shuffle=False,
            args=args
        )
        return [train_loader, val_loader]
