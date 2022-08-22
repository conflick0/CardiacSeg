from monai.transforms import (
    AddChanneld,
    Compose,
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

from transforms.transform import Transform


class MMWHSTransform(Transform):
    def __init__(self, args):
        super().__init__(args)

    def get_train_transform(self):
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

    def get_val_transform(self):
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