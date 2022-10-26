import os
from pathlib import PurePath

from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    LabelFilterd,
    MapLabelValued,
    SaveImaged
)


def get_crop_fg_margin(crop_fg_key):
  if crop_fg_key == "image":
    return 0
  else:
    return 40


def get_train_transform(args):
    margin = get_crop_fg_margin(args.crop_fg_key)
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            LabelFilterd(keys=["label"], applied_labels=[2]),
            MapLabelValued(keys=["label"], orig_labels=[0, 2], target_labels=[0, 1]),
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
            CropForegroundd(
              keys=['image', 'label'], 
              source_key=args.crop_fg_key,
              margin=margin
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
    margin = get_crop_fg_margin(args.crop_fg_key)
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            LabelFilterd(keys=["label"], applied_labels=[2]),
            MapLabelValued(keys=["label"], orig_labels=[0, 2], target_labels=[0, 1]),
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
            CropForegroundd(
              keys=['image', 'label'], 
              source_key=args.crop_fg_key,
              margin=margin
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def save_transform_lbl(data_dict, dst_file_path, print_log=False):
    '''save tansform lbl file and rename SaveImaged default prefix file name'''
    src_file_name = PurePath(data_dict['label']).parts[-1]
    dst_dir = os.path.join(*PurePath(dst_file_path).parts[:-1])
    default_dst_pth = os.path.join(dst_dir, src_file_name)

    tf = Compose(
        [
            LoadImaged(keys=["label"]),
            LabelFilterd(keys=["label"], applied_labels=[2]),
            MapLabelValued(keys=["label"], orig_labels=[0, 2], target_labels=[0, 1]),
            SaveImaged(keys=['label'], output_dir=dst_dir, output_postfix='', separate_folder=False, print_log=False)
        ]
    )

    tf(data_dict)

    os.rename(default_dst_pth, dst_file_path)

    if print_log:
        print(f'save file to {dst_file_path}')

