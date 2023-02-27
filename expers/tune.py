import sys
# set package path
sys.path.append("/nfs/Workspace/CardiacSeg")

import os
from functools import partial

import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import air, tune
from ray.tune import CLIReporter

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from expers.args import get_parser, map_args_transform, map_args_optim, map_args_lrschedule
from data_utils.dataset import DataLoader, get_label_names
from data_utils.utils import get_pids_by_loader
from runners.tuner import run_training
from runners.tester import run_testing
from networks.network import network
from optimizers.optimizer import Optimizer, LR_Scheduler


def main(config, args=None):
    if args.tune_mode == 'transform':
        args = map_args_transform(config, args)
    elif args.tune_mode == 'optim':
        args = map_args_transform(config['transform'], args)
        args = map_args_optim(config['optim'], args)
    elif args.tune_mode == 'lrschedule' or args.tune_mode == 'lrschedule_epoch':
        args = map_args_transform(config['transform'], args)
        args = map_args_optim(config['optim'], args)
        args = map_args_lrschedule(config['lrschedule'], args)
    else:
        # for LinearWarmupCosineAnnealingLR
        args.max_epochs = args.max_epoch
        print('a_max', args.a_max)
        print('a_min', args.a_min)
        print('space_x', args.space_x)
        print('roi_x', args.roi_x)
        print('lr', args.lr)
        print('weight_decay', args.weight_decay)
        print('warmup_epochs',args.warmup_epochs)
        print('max_epochs',args.max_epochs)
    
    
    # train
    args.test_mode = False
    args.checkpoint = os.path.join(args.model_dir, 'final_model.pth')
    main_worker(args)
    # test
    args.test_mode = True
    args.checkpoint = os.path.join(args.model_dir, 'best_model.pth')
    main_worker(args)
    


def main_worker(args):
    # # make dir
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # device
    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")

    # load train and test data
    loader = DataLoader(args.data_name, args)()

    # model
    model = network(args.model_name, args)

    # loss
    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    
    # optimizer
    print(f'optimzer: {args.optim}')
    optimizer = Optimizer(args.optim, model.parameters(), args)

    # lrschedule
    if args.lrschedule is not None:
        print(f'lrschedule: {args.lrschedule}')
        scheduler = LR_Scheduler(args.lrschedule, optimizer, args)
    else:
        scheduler = None

    # check point
    start_epoch = args.start_epoch
    early_stop_count = args.early_stop_count
    best_acc = 0
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # load lrschedule
        if args.lrschedule is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        # load check point epoch and best acc
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "early_stop_count" in checkpoint:
            early_stop_count = checkpoint["early_stop_count"]
        print(
          "=> loaded checkpoint '{}' (epoch {}) (bestacc {}) (early stop count {})"\
          .format(args.checkpoint, start_epoch, best_acc, early_stop_count)
        )

    # inferer
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    # writer
    writer = SummaryWriter(log_dir=args.log_dir)

    if not args.test_mode:
        tr_loader, val_loader = loader
        
        # training
        run_training(
            start_epoch=start_epoch,
            best_acc=best_acc,
            early_stop_count=early_stop_count,
            model=model,
            train_loader=tr_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_func=dice_loss,
            acc_func=dice_acc,
            model_inferer=model_inferer,
            post_label=post_label,
            post_pred=post_pred,
            writer=writer,
            args=args,
        )
    
    else:
        os.makedirs(args.eval_dir, exist_ok=True)
        
        tt_loader = loader[0]

        # test
        dc_vals, hd95_vals = run_testing(
            model,
            tt_loader,
            model_inferer,
            post_label,
            post_pred
        )

        pids = get_pids_by_loader(tt_loader)
        
        label_names = get_label_names(args.data_name)
        
        eval_tt_dice_val_df = pd.DataFrame(
            dc_vals,
            columns=[f'dice{n}' for n in label_names]
        )
        eval_tt_hd95_val_df = pd.DataFrame(
            hd95_vals,
            columns=[f'hd95{n}' for n in label_names]
        )
        eval_tt_df = pd.DataFrame({
            'patientId': pids,
            'type': 'test',
        })

        eval_tt_df = pd.concat([eval_tt_df, eval_tt_dice_val_df, eval_tt_hd95_val_df], axis=1, join='inner')\
            .reset_index(drop=True)
        eval_tt_df.to_csv(os.path.join(args.eval_dir, f'best_model_eval.csv'), index=False)
        
        avg_dice = eval_tt_dice_val_df.T.mean().mean()
        avg_hd95 =  eval_tt_hd95_val_df.T.mean().mean()

        print("\neval result:")
        print('avg dice:', avg_dice)
        print('avg hd95:', avg_hd95)
        print(eval_tt_df.to_string())
        
        tune.report(
            tt_dice=avg_dice,
            tt_hd95=avg_hd95,
            val_bst_acc=best_acc, 
            esc=early_stop_count,
        )



if __name__ == "__main__":
    args = get_parser(sys.argv[1:])
    
    if args.tune_mode == 'test':
        print('test mode')
    elif args.tune_mode == 'train':
        search_space = {
            "exp": tune.grid_search([
              {
                  'exp': args.exp_name,
              }
            ])
        }
    elif args.tune_mode == 'transform':
        search_space = {
            'intensity': tune.grid_search([
                [-42, 423], 
                [13, 320], 
                [32, 294]
            ]),
            'space': tune.grid_search([
                [0.76,0.76,1.0],
                [1.0,1.0,1.0]
            ]),
            'roi': tune.grid_search([
                [96,96,96],
                [128,128,128],
            ]),
        }
    elif args.tune_mode == 'optim':
        search_space = {
            'transform': tune.grid_search([
                {
                    'intensity': [-42,423],
                    'space': [1.0,1.0,1.0],
                    'roi':[128,128,128],
                }
            ]),
            'optim': tune.grid_search([
                {'lr':1e-5, 'weight_decay': 3e-3},
                {'lr':1e-5, 'weight_decay': 1e-3},
                {'lr':1e-5, 'weight_decay': 5e-2},
                {'lr':1e-6, 'weight_decay': 1e-3},
            ]),
        }
    elif args.tune_mode == 'lrschedule':
        search_space = {
            'transform': tune.grid_search([
                {
                    'intensity': [-42,423],
                    'space': [1.0,1.0,1.0],
                    'roi':[128,128,128],
                }
            ]),
            'optim': tune.grid_search([
                {'lr':1e-4, 'weight_decay': 5e-4},
                {'lr':5e-4, 'weight_decay': 5e-4},
                {'lr':1e-3, 'weight_decay': 5e-3},
                {'lr':5e-3, 'weight_decay': 5e-3},
            ]),
            'lrschedule': tune.grid_search([
                {'warmup_epochs':20,'max_epoch':1200},
                {'warmup_epochs':40,'max_epoch':1200},
                {'warmup_epochs':20,'max_epoch':1200},
            ])
        }
    elif args.tune_mode == 'lrschedule_epoch':
        search_space = {
            'transform': tune.grid_search([
                {
                    'intensity': [-42,423],
                    'space': [1.0,1.0,1.0],
                    'roi':[128,128,128],
                }
            ]),
            'optim': tune.grid_search([
                {'lr':1e-2, 'weight_decay': 3e-5},
                {'lr':5e-3, 'weight_decay': 5e-4},
                {'lr':5e-4, 'weight_decay': 5e-5},
            ]),
            'lrschedule': tune.grid_search([
                {'warmup_epochs':40,'max_epoch':700},
                {'warmup_epochs':60,'max_epoch':700},
            ])
        }
    else:
        raise ValueError(f"Invalid args tune mode:{args.tune_mode}")

    trainable_with_cpu_gpu = tune.with_resources(partial(main, args=args), {"cpu": 2, "gpu": 1})
    
    if args.resume_tuner:
        print(f'resume tuner form {args.root_exp_dir}')
        restored_tuner = tune.Tuner.restore(os.path.join(args.root_exp_dir, args.exp_name))
        result = restored_tuner.fit()
        
        # for manual test
        if args.tune_mode == 'test':
            print('run test mode ...')
            # get best model path
            result_grid = restored_tuner.get_results()
            best_result = result_grid.get_best_result(metric="tt_dice", mode="max")
            model_pth = os.path.join( best_result.log_dir, 'models', 'best_model.pth')
            # test
            # for LinearWarmupCosineAnnealingLR
            args.max_epochs = args.max_epoch
            args.test_mode = True
            args.checkpoint = os.path.join(model_pth)
            main_worker(args)
    else:
        tuner = tune.Tuner(
            trainable_with_cpu_gpu,
            param_space=search_space,
            run_config=air.RunConfig(
                name=args.exp_name,
                local_dir=args.root_exp_dir
            )
        )
        tuner.fit()
    
