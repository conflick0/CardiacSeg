from monai.networks.nets import SwinUNETR, UNETR, UNet, AttentionUnet, VNet
from networks.cotr.network_architecture.ResTranUnet import ResTranUnet as CoTr
from networks.unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
from networks.uxnet.networks.UXNet_3D.network_backbone import UXNET
from networks.unest.scripts.networks.unest import UNesT

from networks.densevoxnet.DenseVoxelNet import DenseVoxelNet
from networks.transunet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.transunet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from networks.networkx.unetcnx import UNETCNX
from networks.networkx.unetcnx_x1 import UNETCNX_X1
from networks.networkx.unetcnx_x0 import UNETCNX_X0
from networks.networkx.unetcnx_x2 import UNETCNX_X2 
from networks.networkx.unetcnx_x2_1 import UNETCNX_X2_1
from networks.networkx.unetcnx_x2_2 import UNETCNX_X2_2
from networks.networkx.unetcnx_x3 import UNETCNX_X3
from networks.networkx.unetcnx_x3_1 import UNETCNX_X3_1
from networks.networkx.unetcnx_x3_2 import UNETCNX_X3_2
from networks.networkx.unetcnx_x3_2_2 import UNETCNX_X3_2_2
from networks.networkx.unetcnx_x3_2_2_a1 import UNETCNX_X3_2_2_A1
from networks.networkx.unetcnx_x3_2_2_a2 import UNETCNX_X3_2_2_A2
from networks.networkx.unetcnx_x3_2_2_a3 import UNETCNX_X3_2_2_A3
from networks.networkx.unetcnx_x3_2_2_a4 import UNETCNX_X3_2_2_A4
from networks.networkx.unetcnx_x3_2_2_a5 import UNETCNX_X3_2_2_A5
from networks.networkx.unetcnx_x3_2_2_a6 import UNETCNX_X3_2_2_A6
from networks.networkx.unetcnx_x3_2_2_a7 import UNETCNX_X3_2_2_A7
from networks.networkx.unetcnx_x3_2_2_1 import UNETCNX_X3_2_2_1
from networks.networkx.unetcnx_x4 import UNETCNX_X4
from networks.networkx.unetcnx_x4_1 import UNETCNX_X4_1
from networks.networkx.unetcnx_x4_2 import UNETCNX_X4_2
from networks.networkx.unetcnx_x5 import UNETCNX_X5
from networks.networkx.unetcnx_x6 import UNETCNX_X6
from networks.networkx.unetcnx_x6_1 import UNETCNX_X6_1

from networks.networkx.unetcnx_a0 import UNETCNX_A0
from networks.networkx.unetcnx_a0_s0 import UNETCNX_A0_S0
from networks.networkx.unetcnx_a0_s1 import UNETCNX_A0_S1
from networks.networkx.unetcnx_a0_s2 import UNETCNX_A0_S2
from networks.networkx.unetcnx_a0_s3 import UNETCNX_A0_S3
from networks.networkx.unetcnx_a1 import UNETCNX_A1
from networks.networkx.unetcnx_a2 import UNETCNX_A2
from networks.networkx.unetcnx_a3 import UNETCNX_A3
from networks.networkx.unetcnx_a4 import UNETCNX_A4
from networks.networkx.unetcnx_a4_s0 import UNETCNX_A4_S0
from networks.networkx.unetcnx_a4_s1 import UNETCNX_A4_S1
from networks.networkx.unetcnx_a4_s0_b0 import UNETCNX_A4_S0_B0
from networks.networkx.unetcnx_a5 import UNETCNX_A5

from networks.networkx.unetmnx import UNETMNX
from networks.networkx.unetmnx_a1 import UNETMNX_A1
from networks.networkx.unetmnx_a2 import UNETMNX_A2

from networks.networkx.unetc2f import UNETC2F
from networks.networkx.unetfct import UNETFCT

from networks.networkx.unetcst_a0 import UNETCST_A0
from networks.networkx.unetcst_a1 import UNETCST_A1
from networks.networkx.unetcst_a2 import UNETCST_A2
from networks.networkx.unetcst_a3 import UNETCST_A3
from networks.networkx.unetcst_a4 import UNETCST_A4
from networks.networkx.unetcst_a5 import UNETCST_A5
from networks.networkx.unetcst_a6 import UNETCST_A6
from networks.networkx.unetcst_a7 import UNETCST_A7
from networks.networkx.unetcst_a8 import UNETCST_A8
from networks.networkx.unetcst_a9 import UNETCST_A9
from networks.networkx.unetcst_a9_1 import UNETCST_A9_1


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
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24
          ).to(args.device)
    elif model_name == 'unetcnx_a2':
        return UNETCNX_A2(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=args.feature_size,
              use_init_weights=args.use_init_weights 
          ).to(args.device)
    elif model_name == 'unetcnx_a3':
        return UNETCNX_A3(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=args.feature_size,
              use_init_weights=args.use_init_weights 
          ).to(args.device)
    elif model_name == 'unetcnx_a4':
        return UNETCNX_A4(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            patch_size=args.patch_size,
            stochastic_depth_prob=args.drop_rate,
            depths=args.depths,
            feature_size=args.feature_size,
            kernel_size=args.kernel_size,
            exp_rate=args.exp_rate,
            norm_name=args.norm_name,
            use_init_weights=args.use_init_weights 
        ).to(args.device)
    elif model_name == 'unetcnx_a4_s0':
        return UNETCNX_A4_S0(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            stochastic_depth_prob=args.drop_rate,
            depths=args.depths,
            feature_size=args.feature_size,
            kernel_size=args.kernel_size,
            exp_rate=args.exp_rate,
            norm_name=args.norm_name,
            use_init_weights=args.use_init_weights 
        ).to(args.device)
    elif model_name == 'unetcnx_a4_s1':
        return UNETCNX_A4_S1(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            stochastic_depth_prob=args.drop_rate,
            depths=args.depths,
            feature_size=args.feature_size,
            kernel_size=args.kernel_size,
            exp_rate=args.exp_rate,
            norm_name=args.norm_name,
            use_init_weights=args.use_init_weights 
        ).to(args.device)
    elif model_name == 'unetcnx_a4_s0_b0':
        return UNETCNX_A4_S0_B0(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            patch_size=args.patch_size,
            stochastic_depth_prob=args.drop_rate,
            depths=args.depths,
            feature_size=args.feature_size,
            kernel_size=args.kernel_size,
            exp_rate=args.exp_rate,
            norm_name=args.norm_name,
            use_init_weights=args.use_init_weights 
        ).to(args.device)
    elif model_name == 'unetcnx_a5':
        return UNETCNX_A5(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            patch_size=args.patch_size,
            stochastic_depth_prob=args.drop_rate,
            depths=args.depths,
            feature_size=args.feature_size,
            kernel_size=args.kernel_size,
            exp_rate=args.exp_rate,
            norm_name=args.norm_name,
            use_init_weights=args.use_init_weights 
        ).to(args.device)
    elif model_name == 'unetcst_a9_1':
        return UNETCST_A9_1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcst_a9':
        return UNETCST_A9(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcst_a8':
        return UNETCST_A8(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcst_a7':
        return UNETCST_A7(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcst_a6':
        return UNETCST_A6(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcst_a5':
        return UNETCST_A5(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcst_a4':
        return UNETCST_A4(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcst_a3':
        return UNETCST_A3(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcst_a2':
        return UNETCST_A2(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcst_a1':
        return UNETCST_A1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetfct':
        return UNETFCT(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetc2f':
        return UNETC2F(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcst_a0':
        return UNETCST_A0(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcnx_a0_s0':
        return UNETCNX_A0_S0(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcnx_a0_s1':
        return UNETCNX_A0_S1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcnx_a0_s2':
        return UNETCNX_A0_S2(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcnx_a0_s3':
        return UNETCNX_A0_S3(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcnx_a0':
        return UNETCNX_A0(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
              use_init_weights=args.use_init_weights
          ).to(args.device)
    
    elif model_name == 'unetcnx':
        return UNETCNX(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            patch_size=4
        ).to(args.device)
    
    elif model_name == 'unetcnx_x0':
        return UNETCNX_X0(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=24,
            patch_size=2
        ).to(args.device)

    elif model_name == 'unetcnx_x2':
        return UNETCNX_X2(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=24,
            patch_size=2
        ).to(args.device)

    elif model_name == 'unetcnx_x2_1':
        return UNETCNX_X2_1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcnx_x2_2':
        return UNETCNX_X2_2(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    
    elif model_name == 'unetcnx_x3':
        return UNETCNX_X3(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_1':
        return UNETCNX_X3_1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2':
        return UNETCNX_X3_2(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2_2':
        return UNETCNX_X3_2_2(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2_2_a1':
        return UNETCNX_X3_2_2_A1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2_2_a2':
        return UNETCNX_X3_2_2_A2(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2_2_a3':
        return UNETCNX_X3_2_2_A3(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2_2_a4':
        return UNETCNX_X3_2_2_A4(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2_2_a5':
        return UNETCNX_X3_2_2_A5(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2_2_a6':
        return UNETCNX_X3_2_2_A6(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x3_2_2_a7':
        return UNETCNX_X3_2_2_A7(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              patch_size=args.patch_size,
              stochastic_depth_prob=args.drop_rate,
              depths=args.depths,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x4':
        return UNETCNX_X4(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x4_1':
        return UNETCNX_X4_1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x4_2':
        return UNETCNX_X4_2(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x5':
        return UNETCNX_X5(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x6':
        return UNETCNX_X6(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x6_1':
        return UNETCNX_X6_1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
          ).to(args.device)
    elif model_name == 'unetcnx_x1':
        return UNETCNX_X1(
              in_channels=args.in_channels,
              out_channels=args.out_channels,
              feature_size=24,
              patch_size=2
        ).to(args.device)
    
        
    # -----------------------------------------------------------------------------------------------------
    # unetmnx exp netowrks
    # -----------------------------------------------------------------------------------------------------
    elif model_name == 'unetmnx':
        return UNETMNX(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=24
        ).to(args.device)
    elif model_name == 'unetmnx_a1':
        return UNETMNX_A1(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=24
        ).to(args.device)
    elif model_name == 'unetmnx_a2':
        return UNETMNX_A2(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=24
        ).to(args.device)
    
    else:
        raise ValueError(f'not found model name: {model_name}')

