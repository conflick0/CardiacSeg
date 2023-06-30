import os
import importlib
from pathlib import PurePath

import pandas as pd
from ray import tune

from datasets.chgh_dataset import get_data_dicts
from data_utils.utils import get_pids_by_data_dicts


def get_tune_model_dir(root_exp_dir, exp_name):
    experiment_path = os.path.join(root_exp_dir, exp_name)

    print(f"Loading results from {experiment_path}...")

    restored_tuner = tune.Tuner.restore(experiment_path)
    result_grid = restored_tuner.get_results()

    best_result = result_grid.get_best_result(metric="tt_dice", mode="max")
    print( f"\nBest trial {best_result.metrics['trial_id']}: ")
    print('config:', best_result.metrics['config'])
    print('tt_dice:', best_result.metrics['tt_dice'])
    print('tt_hd95:', best_result.metrics['tt_hd95'])
    if 'esc' in best_result.metrics: 
        print('esc:', best_result.metrics['esc'])
    print(f'best log dir:', best_result.log_dir)
    model_dir = os.path.join( best_result.log_dir, 'models')
    return model_dir


def get_tune_dir(exp_dir):
    restored_tuner = tune.Tuner.restore(exp_dir)
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric="tt_dice", mode="max")
    return best_result.log_dir


def get_data_path(data_dir, data_name, pid):
    dataset = importlib.import_module(f'datasets.{data_name}_dataset')
    _get_data_dicts = getattr(dataset, 'get_data_dicts', None)
    data_dicts = _get_data_dicts(data_dir)
    pids = get_pids_by_data_dicts(data_dicts)
    idx = pids.index(pid)
    return data_dicts[idx]


def get_pred_path(root_dir, exp_name, img_pth):
    return os.path.join(
        root_dir,
        'infers',
        exp_name,
        PurePath(img_pth).parts[-1]
    )


def get_eval_csv_path(root_dir):
    return os.path.join(
        root_dir, 
        'evals',
        'best_model.csv'
    )


def get_eda_eval_csv_path(root_dir):
    return os.path.join(
        root_dir, 
        'eda_evals',
        'best_model.csv'
    )

def get_eval_val(csv_pth, pid):
    df = pd.read_csv(csv_pth)
    idx = df['patientId'] == pid
    if 'inf_diceLV' in df[idx]:
        return {
            'dice': df[idx].filter(regex=("inf_dice*")).T.mean().tolist()[0], 
            'hd95': df[idx].filter(regex=("inf_hd*")).T.mean().tolist()[0]
        }
    else:
        return {
            'dice': df[idx]['inf_diceC'].tolist()[0], 
            'hd95': df[idx]['inf_hd95C'].tolist()[0]
        }

def get_slice(img, slice_idx, mode, is_trans):
    '''
    mode: a, c, s
    '''
    if mode == 'a':
        img = img[:, :, slice_idx]
    elif mode == 's':
        img = img[:, slice_idx, :]
    else:
        img = img[slice_idx, :, :]
    
    if is_trans:
        return img.T
    else:
        return img


def get_img_lbl_preds(data, exp_names, slice_idxs, mode='a', is_trans=False):
    '''
    mode: a, c, s
    '''
    preds = []
    imgs = []
    lbls = []
    for slice_idx in slice_idxs:
        pred_ls = []
        for exp_name in exp_names:
              pred_ls.append(get_slice(data[exp_name], slice_idx, mode, is_trans))
        preds.append(pred_ls)
        imgs.append(get_slice(data['image'], slice_idx, mode, is_trans))
        lbls.append(get_slice(data['label'], slice_idx, mode, is_trans))
    return imgs, lbls, preds