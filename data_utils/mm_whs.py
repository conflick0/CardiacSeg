import os
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
    img_fns = sorted(os.listdir(data_dir))[::2]
    lbl_fns = sorted(os.listdir(data_dir))[1::2]
    img_pths = list(map(lambda fn: os.path.join(data_dir, fn), img_fns))
    lbl_pths = list(map(lambda fn: os.path.join(data_dir, fn), lbl_fns))
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
            LabelFilterd(keys=["label"], applied_labels=[500, 600, 420, 550, 205, 820, 850]),
            MapLabelValued(keys=["label"], orig_labels=[0, 500, 600, 420, 550, 205, 820, 850],
                           target_labels=[0, 1, 2, 3, 4, 5, 6, 7]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.78, 0.78, 1),
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
            LabelFilterd(keys=["label"], applied_labels=[500, 600, 420, 550, 205, 820, 850]),
            MapLabelValued(keys=["label"], orig_labels=[0, 500, 600, 420, 550, 205, 820, 850],
                           target_labels=[0, 1, 2, 3, 4, 5, 6, 7]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.78, 0.78, 1),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
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


def get_loader(args):
    data_dicts = get_data_dicts(args.data_dir)
    train_files, val_files, test_files = data_dicts[:13], data_dicts[13:16], data_dicts[16:]

    if args.test_mode:
        val_transform = get_val_transform()
        test_loader = get_data_loader(
            files=test_files,
            transform=val_transform,
            shuffle=False,
            args=args
        )
        return test_loader
    else:
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