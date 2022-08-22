import os
from pathlib import PurePath

from data_utils.dataset import Dataset
from data_utils.data_loader import MyDataLoader, split_dataset
from transforms.segthor_transform import SegTHORTransform


class SegTHORDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_data_paths(self):
        img_pths = []
        lbl_pths = []
        for patient_dir in sorted(os.listdir(self.data_dir)):
            img_pths.append(os.path.join(self.data_dir, patient_dir, f'{patient_dir}.nii.gz'))
            lbl_pths.append(os.path.join(self.data_dir, patient_dir, f'GT.nii.gz'))
        return img_pths, lbl_pths


def get_test_pids(data_dir):
    ds = SegTHORDataset(data_dir)
    train_files, val_files, test_files = split_dataset(ds.get_data_dicts())
    pids = list(map(lambda x: PurePath(x['image']).parts[-1].split('.')[0], test_files))
    return pids


def get_loader(args):
    ds = SegTHORDataset(args.data_dir)
    tf = SegTHORTransform(args)

    data_dicts = ds.get_data_dicts()
    train_transform = tf.get_train_transform()
    val_transform = tf.get_val_transform()

    dl = MyDataLoader(
        data_dicts,
        train_transform,
        val_transform,
        args
    )

    return dl.get_loader()
