from pathlib import PurePath


def get_pids_by_files(files):
    pids = list(map(lambda x: PurePath(x).parts[-1].split('.')[0], files))
    return pids


def get_pids_by_data_dicts(data_dicts):
    files = list(map(lambda x: x['image'], data_dicts))
    return get_pids_by_files(files)


def get_pids_by_loader(loader):
    files = [data['image_meta_dict']['filename_or_obj'][0] for data in loader]
    return get_pids_by_files(files)
