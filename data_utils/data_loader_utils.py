import os
import math

from sklearn.model_selection import KFold

from data_utils.io import load_json
from data_utils.utils import get_pids_by_data_dicts


def split_data_dicts(data_dicts, fold, split_train_ratio, num_fold):
    '''split data to train, val and test'''
    num_data = len(data_dicts)
    num_train_data = math.ceil(num_data * split_train_ratio)

    train_val_idxs = list(range(0, num_train_data))
    test_idxs = list(range(num_train_data, num_data))

    kf = KFold(n_splits=num_fold)
    flods = list(kf.split(train_val_idxs))
    train_idxs, val_idxs = flods[fold]

    train_files = [data_dicts[i] for i in train_idxs]
    val_files = [data_dicts[i] for i in val_idxs]
    test_files = [data_dicts[i] for i in test_idxs]

    print(f'fold: {fold}')
    print(f"train files ({len(train_idxs)}):", get_pids_by_data_dicts(train_files))
    print(f"val files ({len(val_idxs)}):", get_pids_by_data_dicts(val_files))
    print(f"test files ({len(test_idxs)}):", get_pids_by_data_dicts(test_files))

    return train_files, val_files, test_files


def get_abs_data_dicts(data_dir, data_dicts):
    '''get absolute path data dicts'''
    out_data_dicts = []
    for data_dict in data_dicts:
        out_data_dict = {}
        for k in data_dict.keys():
            out_data_dict[k] = os.path.join(data_dir, data_dict[k])
        out_data_dicts.append(out_data_dict)
    return out_data_dicts


def load_data_dict_json(data_dir, data_dict_json):
    data_dicts = load_json(data_dict_json)
    train_files = get_abs_data_dicts(
        data_dir,
        data_dicts.get('train', None)
    )
    val_files = get_abs_data_dicts(
        data_dir,
        data_dicts.get('val', None)
    )
    test_files = get_abs_data_dicts(
        data_dir,
        data_dicts.get('test', None)
    )
    print(f"train files ({len(train_files)}):", get_pids_by_data_dicts(train_files))
    print(f"val files ({len(val_files)}):", get_pids_by_data_dicts(val_files))
    print(f"test files ({len(test_files)}):", get_pids_by_data_dicts(test_files))
    return train_files, val_files, test_files