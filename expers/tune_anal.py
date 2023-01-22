import argparse
import os

import torch
from ray import tune


parser = argparse.ArgumentParser(description="model segmentation pipeline")

parser.add_argument("--exp_name", default="", type=str, help="exp name")
parser.add_argument("--local_dir", default="", type=str, help="tune directory")

args = parser.parse_args()

experiment_path = os.path.join(args.local_dir, args.exp_name)

print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path)
result_grid = restored_tuner.get_results()

for i, result in enumerate(result_grid):
    if result.error:
        print(f"Trial #{i} had an error:", result.error)
        continue

    print(
        f"Trial {result.metrics['trial_id']}: ",
        result.metrics['config'],
        result.metrics['tt_dice'] ,
    )
    
best_result = result_grid.get_best_result(metric="tt_dice", mode="max")
print( f"\nBest trial {best_result.metrics['trial_id']}: ")
print('config:', best_result.metrics['config'])
print('tt_dice:', best_result.metrics['tt_dice'])
print('tt_hd95:', best_result.metrics['tt_hd95'])
print('esc:', best_result.metrics['esc'])
print(f'best log dir:', best_result.log_dir)


model_pth = os.path.join( best_result.log_dir,'models', 'final_model.pth')
ckp = torch.load(model_pth)
print('final early stop count:', ckp['early_stop_count'])
print('final epoch:', ckp['epoch'])