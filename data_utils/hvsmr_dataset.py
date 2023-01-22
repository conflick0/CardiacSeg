import os

from data_utils.data_loader import MyDataLoader, split_data_dicts, get_dl
from transforms.hvsmr_transform import get_train_transform, get_val_transform


def get_data_dicts(data_dir):
    data_dicts = []
    fns = sorted(os.listdir(data_dir))
    for img_fn, lbl_fn in zip(fns[::3], fns[1::3]):
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, img_fn)),
            "label": os.path.join(os.path.join(data_dir, lbl_fn))
        })
    return data_dicts


def get_loader(args):
    data_dicts = get_data_dicts(args.data_dir)
    train_transform = get_train_transform(args)
    val_transform = get_val_transform(args)

    dl = MyDataLoader(
        data_dicts,
        train_transform,
        val_transform,
        args
    )

    return dl.get_loader()


def get_infer_loader(args):
    data_dicts = [{'image': args.img_pth, 'label': args.lbl_pth}]

    val_transform = get_val_transform(args)
    inf_loader = get_dl(
        files=data_dicts,
        transform=val_transform,
        shuffle=False,
        args=args
    )
    return inf_loader