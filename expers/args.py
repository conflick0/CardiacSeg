import argparse

def get_parser(argv):
    parser = argparse.ArgumentParser(description="model segmentation pipeline")

    # mode
    parser.add_argument("--tune_mode", default=None, type=str, help="tune mode")
    parser.add_argument("--test_mode", action="store_true", help="test mode")
    parser.add_argument("--save_eval_csv", action="store_true", help="save eval csv")
    parser.add_argument("--resume_tuner", action="store_true", help="test mode")

    # dir and path
    parser.add_argument("--exp_name", default=None, type=str, help="exp name")
    parser.add_argument("--data_name", default=None, type=str, help="exp name")
    parser.add_argument("--root_exp_dir", default=None, type=str, help="root exp directory")
    parser.add_argument("--data_dir", default=None, type=str, help="dataset directory")
    parser.add_argument("--model_dir", default="./models", type=str, help="directory to save the models")
    parser.add_argument("--log_dir", default="./logs", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--eval_dir", default="./evals", type=str, help="directory to save the eval result")
    parser.add_argument("--infer_dir", default="./infers", type=str, help="directory to save the eval result")
    parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
    parser.add_argument("--ssl_checkpoint", default=None, help="start training from saved checkpoint")
    parser.add_argument("--filename", default="best_model.pth", help="save model file name")
    parser.add_argument("--ssl_pretrained", default=None, type=str, help="use self-supervised pretrained weights")
    parser.add_argument("--img_pth", default=None, help="target img for infer")
    parser.add_argument("--lbl_pth", default=None, help="target lbl for infer")
    parser.add_argument("--pred_pth", default=None, help="target lbl for infer")
    parser.add_argument("--eda_test_data", default=None, type=str, help="test train data or val data for eda (train, val), default is test data")

    # train loop
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--val_every", default=20, type=int, help="validation frequency")
    parser.add_argument("--max_epoch", default=2000, type=int, help="max number of training epochs")
    parser.add_argument("--early_stop_count", default=0, type=int, help="early stop count")
    parser.add_argument("--max_early_stop_count", default=20, type=int, help="max early stop count")
    parser.add_argument("--save_checkpoint_freq", default=1, type=int, help="save final checkpoint freq, if value is 0 won't save.")

    # data
    parser.add_argument("--data_dicts_json", default=None, type=str, help="data dicts json")
    parser.add_argument("--split_train_ratio", default=0.75, type=float, help="split train ratio")
    parser.add_argument("--num_fold", default=3, type=int, help="num fold")
    parser.add_argument("--fold", default=2, type=int, help="index of fold")

    # data loader
    parser.add_argument("--data_loader", default='cache', type=str, help="cache dataset, generic dataset")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--pin_memory", action="store_true", help="pin memory")
    parser.add_argument("--workers", default=2, type=int, help="number of workers")

    # transform
    parser.add_argument("--num_samples", default=2, type=int, help="number of samples")
    parser.add_argument("--a_min", default=-1000.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--rand_flipd_prob", default=0.1, type=float, help="RandFlipd aug probability")
    parser.add_argument("--rand_rotate90d_prob", default=0.1, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--rand_shift_intensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--rand_scale_intensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")

    # model
    parser.add_argument("--model_name", default=None, type=str, help="model name")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
    parser.add_argument("--patch_size", default=4, type=int, help="number of patch size")
    parser.add_argument("--feature_size", default=48, type=int, help="number of feature size")
    parser.add_argument("--kernel_size", default=7, type=int, help="number of kernel size")
    parser.add_argument("--exp_rate", default=4, type=int, help="number of exp_rate")
    parser.add_argument("--norm_name", default='layer', type=str, help="norm name")
    parser.add_argument("--drop_rate", default=0.4, type=float, help="drop out rate")
    parser.add_argument('--depths', type=int, nargs='+', help="depths of layer")
    parser.add_argument('--use_init_weights',  action="store_true", help="use init weights")
    parser.add_argument('--is_conv_stem',  action="store_true", help=" is conv stem")
    parser.add_argument("--skip_encoder_name", default=None, type=str, help="skip encoder name")
    parser.add_argument('--deep_sup',  action="store_true", help="use deeply supervised")
    parser.add_argument('--first_feature_size_half',  action="store_true", help="first feature size half")
    
    
    # loss
    parser.add_argument("--loss", default=None, type=str, help="type of loss")
    parser.add_argument("--lambda_dice", default=0.5, type=float, help="lambda of dice")
    parser.add_argument("--lambda_focal", default=0.5, type=float, help="lambda of focal")
    
    # optimizer
    parser.add_argument("--optim", default=None, type=str, help="type of optimizer")
    parser.add_argument("--lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=1e-5, type=float, help="momentum")

    # scheduler
    parser.add_argument("--lrschedule", default=None, type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    parser.add_argument("--power", default=0.9, type=float, help="param of poly")
    parser.add_argument("--last_epoch", default=-1, type=float, help="param of poly")

    # infer
    parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
    parser.add_argument("--infer_overlap", default=0.25, type=float, help="sliding window inference overlap")
    parser.add_argument("--infer_post_process", action="store_true", help="infer post process")

    
    # get args
    args = parser.parse_args(argv)
    
    # total_iters of poly scheduler param 
    args.total_iters = args.max_epoch
    
    return args


def map_args_transform(config, args):
    intensity = config['intensity']
    args.a_min = intensity[0]
    args.a_max = intensity[1]

    search_space = config['space']
    args.space_x = search_space[0]
    args.space_y = search_space[1]
    args.space_z = search_space[2]

    search_space = config['roi']
    args.roi_x = search_space[0]
    args.roi_y = search_space[1]
    args.roi_z = search_space[2]
    return args


def map_args_optim(config, args):
    args.lr = config['lr']
    args.weight_decay = config['weight_decay']
    return args


def map_args_lrschedule(config, args):
    args.warmup_epochs = config['warmup_epochs']
    args.max_epochs = config['max_epoch'] # for LinearWarmupCosineAnnealingLR
    args.max_epoch = config['max_epoch']
    return args


def map_args_network(config, args):
    args.depths = config['depths']
    return args