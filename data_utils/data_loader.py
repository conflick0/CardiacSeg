from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset
)

from data_utils.data_loader_utils import split_data_dicts, load_data_dict_json


def get_dl(files, transform, shuffle, batch_size, args):
    if args.data_loader == 'cache':
        print("Using MONAI Cache Dataset")
        ds = CacheDataset(
            data=files,
            transform=transform,
            num_workers=args.workers
        )
    else:
        print("Using generic dataset")
        ds = Dataset(
            data=files,
            transform=transform,
        )
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.pin_memory
    )
    return loader


class MyDataLoader:
    def __init__(self, get_data_dicts_fn, train_transform, val_transform, args):
        if args.data_dicts_json:
            self.train_files, self.val_files, self.test_files = load_data_dict_json(args.data_dir, args.data_dicts_json)
        else:
            data_dicts = get_data_dicts_fn(args.data_dir)
            self.train_files, self.val_files, self.test_files = split_data_dicts(
                data_dicts,
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

