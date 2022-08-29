import os

from data_utils.data_loader import MyDataLoader
from transforms.segthor_transform import SegTHORTransform


def get_data_dicts(data_dir):
    data_dicts = []
    for patient_dir in sorted(os.listdir(data_dir)):
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'GT.nii.gz'))
        })
    return data_dicts


def get_loader(args):
    tf = SegTHORTransform(args)

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
