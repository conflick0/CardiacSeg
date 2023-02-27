from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    SqueezeDimd,
    ToTensord
)


def get_train_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            RandSpatialCropSamplesd(
                keys=['image','label'],
                roi_size=(args.roi_x, args.roi_y),
                num_samples=args.num_samples,
                random_center=True, 
                random_size=False
            ),
            RandFlipd(
              keys=["image", "label"],
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
            ToTensord(keys=["image", "label"])
        ]
    )


def get_val_transform(args):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, 
            a_max=args.a_max,
            b_min=args.b_min, 
            b_max=args.b_max,
            clip=True,
        ),
        ScaleIntensityd(keys=["image"], minv=0, maxv=255)
    ])


def get_inf_transform(keys, args):
    if len(keys) == 2:
      mode = ("bilinear", "nearest")
    else:
      mode = ("bilinear")

    return Compose([
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=mode,
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, 
            a_max=args.a_max,
            b_min=args.b_min, 
            b_max=args.b_max,
            clip=True,
        ),
        ScaleIntensityd(keys=["image"], minv=0, maxv=255)
    ])