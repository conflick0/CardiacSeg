import math

from sklearn.model_selection import KFold
from monai.data import (
    DataLoader,
    CacheDataset
)

from data_utils.utils import get_pids_by_data_dicts


def get_dl(files, transform, shuffle, batch_size, args):
    ds = CacheDataset(
        data=files,
        transform=transform,
        num_workers=args.workers
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )
    return loader


def get_files(data_dicts, idxs):
    return [data_dicts[i] for i in idxs]


def split_data_dicts(data_dicts, fold, split_train_ratio, num_fold):
    num_data = len(data_dicts)
    num_train_data = math.ceil(num_data * split_train_ratio)

    train_val_idxs = list(range(0, num_train_data))
    test_idxs = list(range(num_train_data, num_data))

    kf = KFold(n_splits=num_fold)
    flods = list(kf.split(train_val_idxs))
    train_idxs, val_idxs = flods[fold]

    train_files = get_files(data_dicts, train_idxs)
    val_files = get_files(data_dicts, val_idxs)
    test_files = get_files(data_dicts, test_idxs)

    print(f'fold: {fold}')
    print(f"train files ({len(train_idxs)}):", get_pids_by_data_dicts(train_files))
    print(f"val files ({len(val_idxs)}):", get_pids_by_data_dicts(val_files))
    print(f"test files ({len(test_idxs)}):", get_pids_by_data_dicts(test_files))

    return train_files, val_files, test_files


class MyDataLoader:
    def __init__(self, data_dicts, train_transform, val_transform, args):
        self.data_dicts = data_dicts
        self.train_files, self.val_files, self.test_files = split_data_dicts(
          self.data_dicts, 
          args.fold, 
          args.split_train_ratio,
          args.num_fold
        )
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.args = args

    def get_loader(self):
        if self.args.test_mode:
            print('\nload test dataset ...',)
            test_loader = get_dl(
                files=self.test_files,
                transform=self.val_transform,
                shuffle=False,
                batch_size=self.args.batch_size,
                args=self.args
            )
            return [test_loader]
        else:
            print('\nload train dataset ...')
            train_loader = get_dl(
                files=self.train_files,
                transform=self.train_transform,
                shuffle=True,
                batch_size=self.args.batch_size,
                args=self.args
            )
            print('\nload val dataset ...')
            val_loader = get_dl(
                files=self.val_files,
                transform=self.val_transform,
                shuffle=False,
                batch_size=self.args.batch_size,
                args=self.args
            )
            return [train_loader, val_loader]


class MyDataLoader2d:
    def __init__(self, data_dicts, train_transform, val_transform, args):
        self.data_dicts = data_dicts
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.args = args
        self.train_files, self.val_files, self.test_files = self.data_dicts['training'], self.data_dicts['validation'], self.data_dicts['test']

    def get_loader(self):
        if self.args.test_mode:
            print('\nload test dataset ...',)
            test_loader = get_dl(
                files=self.test_files,
                transform=self.val_transform,
                shuffle=False,
                batch_size=self.args.val_batch_size,
                args=self.args
            )
            return [test_loader]
        else:
            print('\nload train dataset ...')
            train_loader = get_dl(
                files=self.train_files,
                transform=self.train_transform,
                shuffle=True,
                batch_size=self.args.batch_size,
                args=self.args
            )
            print('\nload val dataset ...')
            val_loader = get_dl(
                files=self.val_files,
                transform=self.val_transform,
                shuffle=False,
                batch_size=self.args.val_batch_size,
                args=self.args
            )
            return [train_loader, val_loader]
