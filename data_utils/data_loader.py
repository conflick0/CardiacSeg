import math

from monai.data import (
    DataLoader,
    CacheDataset
)


def get_dl(files, transform, shuffle, args):
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


def split_dataset(data_dicts, split_train_ratio=0.8):
    num_data = len(data_dicts)
    num_train_val_data = math.ceil(num_data * split_train_ratio)
    num_train_data = math.ceil(num_train_val_data * split_train_ratio)
    return data_dicts[:num_train_data], data_dicts[num_train_data:num_train_val_data], data_dicts[num_train_val_data:]


class MyDataLoader:
    def __init__(self, data_dicts, train_transform, val_transform, args):
        self.data_dicts = data_dicts
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.args = args

    def get_loader(self):
        train_files, val_files, test_files = split_dataset(self.data_dicts)

        if self.args.test_mode:
            test_loader = get_dl(
                files=test_files,
                transform=self.val_transform,
                shuffle=False,
                args=self.args
            )
            return [test_loader]
        else:
            train_loader = get_dl(
                files=train_files,
                transform=self.train_transform,
                shuffle=True,
                args=self.args
            )
            val_loader = get_dl(
                files=val_files,
                transform=self.val_transform,
                shuffle=False,
                args=self.args
            )
            return [train_loader, val_loader]

