from monai.networks.nets import SwinUNETR, UNETR, UNet, AttentionUnet, VNet
from networks.cotr.network_architecture.ResTranUnet import ResTranUnet as CoTr
from networks.unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from networks.uxnet.networks.UXNet_3D.network_backbone import UXNET
from networks.unest.scripts.networks.unest import UNesT

from networks.densevoxnet.DenseVoxelNet import DenseVoxelNet
from networks.transunet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.transunet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from networks.networkx.unetcnx_a1 import UNETCNX_A1


def network(model_name, args):
    print(f'model: {model_name}')
    if model_name == 'unet3d':
        return UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(64, 128, 256, 256),
            strides=(2, 2, 2),
            num_res_units=0,
            act='RELU',
            norm='BATCH'
        ).to(args.device)

    elif model_name == 'attention_unet':
        return AttentionUnet(
          spatial_dims=3,
          in_channels=args.in_channels,
          out_channels=args.out_channels,
          channels=(32, 64, 128, 256),
          strides=(2, 2, 2),
        ).to(args.device)
    
    elif model_name == 'vnet':
        return VNet(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
        ).to(args.device)
    
    elif model_name == 'cotr':
        '''
        CAUTION: if deep_supervision is True mean network output will be 
        a list e.x. [result, ds0, ds1, ds2], so loss func 
        should be use CoTr deep supervision loss.
        '''
        # TODO: deep_supervision 
        return CoTr(
            norm_cfg='IN',
            activation_cfg='LeakyReLU',
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            num_classes=args.out_channels,
            weight_std=False,
            deep_supervision=False
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

    elif model_name == 'swinunetr':
        return SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            use_checkpoint=True,
        ).to(args.device)
    
    
    elif model_name == 'unetr_pp':
        return UNETR_PP(
          in_channels=args.in_channels,
          out_channels=args.out_channels,
          img_size=[args.roi_x, args.roi_y, args.roi_z],
          feature_size=12,
          num_heads=4,
          depths=[3, 3, 3, 3],
          dims=[24, 48, 96, 192],
          do_ds=False,
        ).to(args.device)

    elif model_name == 'uxnet':
        return UXNET(
            in_chans=args.in_channels,
            out_chans=args.out_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        ).to(args.device)
    
    elif model_name == 'unest':
        return UNesT(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels
        ).to(args.device)
    # -----------------------------------------------------------------------------------------------------
    # cardiac segment netowrks
    # -----------------------------------------------------------------------------------------------------
    elif model_name == 'dense_vox_net':
        return DenseVoxelNet(
            in_channels=args.in_channels, 
            classes=args.out_channels
        ).to(args.device)
    
    # -----------------------------------------------------------------------------------------------------
    # 2d medical image segment netowrks
    # -----------------------------------------------------------------------------------------------------
    elif model_name == 'transunet':
        vit_name = 'R50-ViT-B_16'
        img_size = args.roi_x
        vit_patches_size = 16
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = args.out_channels
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        return ViT_seg(
          config_vit, 
          img_size=img_size, 
          num_classes=config_vit.n_classes
        ).to(args.device)
    
    # -----------------------------------------------------------------------------------------------------
    # unetcnx exp netowrks
    # -----------------------------------------------------------------------------------------------------
    elif model_name == 'unetcnx_a1':
        return UNETCNX_A1(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            patch_size=args.patch_size,
            kernel_size=args.kernel_size,
            exp_rate=args.exp_rate,
            feature_size=args.feature_size,
            depths=args.depths,
            drop_path_rate=args.drop_rate,
            use_init_weights=args.use_init_weights,
            is_conv_stem=args.is_conv_stem,
            skip_encoder_name=args.skip_encoder_name,
            deep_sup=args.deep_sup,
            first_feature_size_half=args.first_feature_size_half
          ).to(args.device)
    else:
        raise ValueError(f'not found model name: {model_name}')

