from pathlib import PurePath

from monai.transforms import (
  Compose,
  LoadImaged
)
import pandas as pd

from data_utils.visualization import show_img_lbl


def get_pids_by_files(files):
    pids = list(map(lambda x: PurePath(x).parts[-1].split('.')[0], files))
    return pids


def get_pids_by_data_dicts(data_dicts):
    files = list(map(lambda x: x['image'], data_dicts))
    return get_pids_by_files(files)


def get_pids_by_loader(loader):
    files = [data['image_meta_dict']['filename_or_obj'][0] for data in loader]
    return get_pids_by_files(files)


def get_label_classes(label):
  return label.flatten().unique().numpy()


def get_data_info(data_dicts):
    '''show data info for eda'''
    base_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            LoadImaged(keys=["label"])
        ]
    )
    df = pd.DataFrame()
    for data_dict in data_dicts:
      d = base_transforms(data_dict)
      pid = d['image_meta_dict']['filename_or_obj'].split('/')[-1].split('.')[0]
      file_pth = d['image_meta_dict']['filename_or_obj']
      img_shape = list(d['image_meta_dict']['spatial_shape'])
      img_space = list(d['image_meta_dict']['pixdim'])[1:4]
      lbl_shape = list(d['label_meta_dict']['spatial_shape'])
      lbl_space = list(d['label_meta_dict']['pixdim'])[1:4]
      lbl_ids = get_label_classes(d['label'])

      print('pid:', pid)
      print('file_pth:', file_pth)
      print('img shape:', img_shape)
      print('img spacing:', img_space) 
      print('lbl shape:', lbl_shape)
      print('lbl spacing:', lbl_space)
      print('lbl ids:', lbl_ids)

      slice_idx=int(d['image'].shape[-1]/2)
      show_img_lbl(
          d['image'][:,:,slice_idx],
          d['label'][:,:,slice_idx],
          slice_idx=slice_idx,
          num_classes=len(lbl_ids),
          axis_off=False,
          fig_size=(10, 5)
      )
      df = df.append(
          {    
              'pid': pid,
              'img_shape':img_shape,
              'img_space':img_space,
              'lbl_shape':lbl_shape,
              'lbl_space':lbl_space,
              'lbl_ids': lbl_ids
          },
          ignore_index=True
      )
      print()
    return df

