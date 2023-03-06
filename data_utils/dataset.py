import importlib
from data_utils.data_loader import MyDataLoader, get_dl


class DataLoader:
    def __init__(self, data_name, args):
        self.data_name = data_name
        self.args = args

        # get dataset module get_data_dicts fn
        dataset = importlib.import_module(f'datasets.{data_name}_dataset')
        self.get_data_dicts = getattr(dataset, 'get_data_dicts', None)

        # get transform module get_train/val_transform fn
        transform = importlib.import_module(f'transforms.{data_name}_transform')
        self.get_train_transform = getattr(transform, 'get_train_transform', None)
        self.get_val_transform = getattr(transform, 'get_val_transform', None)

        # init data loader
        self.dl = MyDataLoader(
            self.get_data_dicts,
            self.get_train_transform(self.args),
            self.get_val_transform(self.args),
            self.args
        )

    def __call__(self):
        return self.dl.get_loader()


def get_infer_data(data_dict, args):
    keys = data_dict.keys()
    data_name = args.data_name
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_inf_transform = getattr(transform, 'get_inf_transform', None)
    inf_transform = get_inf_transform(keys, args)
    data = inf_transform(data_dict)
    return data


def get_infer_loader(keys, args):
    data_dicts = [{'image': args.img_pth, 'label': args.lbl_pth}]
    transform = importlib.import_module(f'transforms.{args.data_name}_transform')
    get_inf_transform = getattr(transform, 'get_inf_transform', None)
    inf_transform = get_inf_transform(keys, args)
    inf_loader = get_dl(
        files=data_dicts,
        transform=inf_transform,
        shuffle=False,
        batch_size=1,
        args=args
    )
    return inf_loader


def get_label_names(data_name):
    label_names_map = {
        'chgh': ['C'],
        'mmwhs': ['LV', 'RV', 'LA', 'RA', 'MLV', 'AA', 'PA'],
        'hvsmr': ['M', 'B'],
        'segthor': ['C']
    }
    return label_names_map[data_name]
