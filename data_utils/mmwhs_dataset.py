import os

from data_utils.data_loader import MyDataLoader
from transforms.mmwhs_transform import MMWHSTransform


def get_data_dicts(data_dir):
    img_fns = sorted(os.listdir(data_dir))[::2]
    lbl_fns = sorted(os.listdir(data_dir))[1::2]
    img_pths = list(map(lambda fn: os.path.join(data_dir, fn), img_fns))
    lbl_pths = list(map(lambda fn: os.path.join(data_dir, fn), lbl_fns))
    data_dicts = []
    for img_pth, lbl_pth in zip(img_pths, lbl_pths):
        data_dicts.append({
            "image": img_pth,
            "label": lbl_pth
        })
    return data_dicts


def get_loader(args):
    tf = MMWHSTransform(args)

    data_dicts = get_data_dicts(args.data_dir)
    train_transform = tf.get_train_transform()
    val_transform = tf.get_val_transform()

    dl = MyDataLoader(
        data_dicts,
        train_transform,
        val_transform,
        args
    )

    return dl.get_loader()
