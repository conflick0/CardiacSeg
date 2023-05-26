from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    LabelFilterd,
    MapLabelValued,
    SqueezeDimd,
)


def get_train_transform(args):
    # CropForegroundd(keys=['image', 'label'], source_key='image'),
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
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=args.rand_flipd_prob,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=args.rand_flipd_prob,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=args.rand_flipd_prob,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=args.rand_rotate90d_prob,
                max_k=3,
            ),
            RandScaleIntensityd(
              keys=["image"], 
              factors=0.1, 
              prob=args.rand_scale_intensityd_prob
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=args.rand_shift_intensityd_prob,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

def get_val_transform(args):
    # CropForegroundd(keys=['image', 'label'], source_key='image'),
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
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_inf_transform(keys, args):
    if len(keys) == 2:
        # image and label
        mode = ("bilinear", "nearest")
        return Compose(
            [
                LoadImaged(keys=keys),
                AddChanneld(keys=keys),
                Orientationd(keys=keys, axcodes="RAS"),
                Spacingd(
                    keys=keys,
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=mode,
                ),
                ScaleIntensityRanged(
                    keys=['image'],
                    a_min=args.a_min, 
                    a_max=args.a_max,
                    b_min=args.b_min, 
                    b_max=args.b_max,
                    clip=True,
                    allow_missing_keys=True
                ),
                LabelFilterd(keys=["label"], applied_labels=[500, 600, 420, 550, 205, 820, 850]),
                MapLabelValued(keys=["label"], orig_labels=[0, 500, 600, 420, 550, 205, 820, 850],
                                target_labels=[0, 1, 2, 3, 4, 5, 6, 7]),
                AddChanneld(keys=keys),
                ToTensord(keys=keys)
            ]
        )

    else:
        # image
        mode = ("bilinear")
        return Compose(
            [
                LoadImaged(keys=keys),
                AddChanneld(keys=keys),
                Orientationd(keys=keys, axcodes="RAS"),
                Spacingd(
                    keys=keys,
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=mode,
                ),
                ScaleIntensityRanged(
                    keys=['image'],
                    a_min=args.a_min, 
                    a_max=args.a_max,
                    b_min=args.b_min, 
                    b_max=args.b_max,
                    clip=True,
                    allow_missing_keys=True
                ),
                AddChanneld(keys=keys),
                ToTensord(keys=keys)
            ]
        )
    

def get_label_transform(keys=["label"]):
    return Compose(
        [
            LoadImaged(keys=keys),
            LabelFilterd(keys=keys, applied_labels=[500, 600, 420, 550, 205, 820, 850]),
            MapLabelValued(keys=keys, orig_labels=[0, 500, 600, 420, 550, 205, 820, 850],
                            target_labels=[0, 1, 2, 3, 4, 5, 6, 7]),
        ]
    )