import os
from pathlib import PurePath

from data_utils.dataset import Dataset
from data_utils.data_loader import MyDataLoader, split_dataset
from transforms.mmwhs_transform import MMWHSTransform


class MMWHSDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_data_paths(self):
        img_fns = sorted(os.listdir(self.data_dir))[::2]
        lbl_fns = sorted(os.listdir(self.data_dir))[1::2]
        img_pths = list(map(lambda fn: os.path.join(self.data_dir, fn), img_fns))
        lbl_pths = list(map(lambda fn: os.path.join(self.data_dir, fn), lbl_fns))
        return img_pths, lbl_pths


def get_test_pids(data_dir):
    ds = MMWHSDataset(data_dir)
    train_files, val_files, test_files = split_dataset(ds.get_data_dicts())
    pids = list(map(lambda x: PurePath(x['image']).parts[-1].split('.')[0], test_files))
    return pids


def get_loader(args):
    ds = MMWHSDataset(args.data_dir)
    tf = MMWHSTransform(args)

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
