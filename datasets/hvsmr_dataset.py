import os


def get_data_dicts(data_dir):
    data_dicts = []
    fns = sorted(os.listdir(data_dir))
    for img_fn, lbl_fn in zip(fns[::3], fns[1::3]):
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, img_fn)),
            "label": os.path.join(os.path.join(data_dir, lbl_fn))
        })
    return data_dicts
