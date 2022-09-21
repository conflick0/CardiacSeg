from monai.networks.nets import SwinUNETR, UNETR

from networks.unetcnx import UNETCNX
from networks.unetcnx_x1 import UNETCNX_X1


def network(model_name, args):
    print(f'model: {model_name}')

    if model_name == 'unetcnx':
        return UNETCNX(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            patch_size=4
        ).to(args.device)

    elif model_name == 'swinunetr':
        return SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            use_checkpoint=True,
        ).to(args.device)

    elif model_name == 'unetr':
        return UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(args.device)

    elif model_name == 'unetcnx_x1':
        return UNETCNX(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=48,
              patch_size=4
        ).to(args.device)

    else:
      raise ValueError(f'not found model name: {model_name}')


