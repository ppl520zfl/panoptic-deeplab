# ------------------------------------------------------------------------------
# Builds model.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
import os
from backbone import resnet, mobilenet, mnasnet
from segmentation.model.nas import fasterseg
from meta_arch import DeepLabV3, DeepLabV3Plus, PanopticDeepLab
from loss import RegularCE, OhemCE, DeepLabCE, L1Loss, MSELoss
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat


def build_segmentation_model_from_cfg(config):
    """Builds segmentation model with specific configuration.
    Args:
        config: the configuration.

    Returns:
        A nn.Module segmentation model.
    """
    model_map = {
        'deeplabv3': DeepLabV3,
        'deeplabv3plus': DeepLabV3Plus,
        'panoptic_deeplab': PanopticDeepLab,
    }

    model_cfg = {
        'deeplabv3': dict(
            replace_stride_with_dilation=config.MODEL.BACKBONE.DILATION,
            in_channels=config.MODEL.DECODER.IN_CHANNELS,
            feature_key=config.MODEL.DECODER.FEATURE_KEY,
            decoder_channels=config.MODEL.DECODER.DECODER_CHANNELS,
            atrous_rates=config.MODEL.DECODER.ATROUS_RATES,
            num_classes=config.DATASET.NUM_CLASSES,
            semantic_loss=build_loss_from_cfg(config.LOSS.SEMANTIC),
            semantic_loss_weight=config.LOSS.SEMANTIC.WEIGHT,
        ),
        'deeplabv3plus': dict(
            replace_stride_with_dilation=config.MODEL.BACKBONE.DILATION,
            in_channels=config.MODEL.DECODER.IN_CHANNELS,
            feature_key=config.MODEL.DECODER.FEATURE_KEY,
            low_level_channels=config.MODEL.DEEPLABV3PLUS.LOW_LEVEL_CHANNELS,
            low_level_key=config.MODEL.DEEPLABV3PLUS.LOW_LEVEL_KEY,
            low_level_channels_project=config.MODEL.DEEPLABV3PLUS.LOW_LEVEL_CHANNELS_PROJECT,
            decoder_channels=config.MODEL.DECODER.DECODER_CHANNELS,
            atrous_rates=config.MODEL.DECODER.ATROUS_RATES,
            num_classes=config.DATASET.NUM_CLASSES,
            semantic_loss=build_loss_from_cfg(config.LOSS.SEMANTIC),
            semantic_loss_weight=config.LOSS.SEMANTIC.WEIGHT,
        ),
        'panoptic_deeplab': dict(
            replace_stride_with_dilation=config.MODEL.BACKBONE.DILATION,
            in_channels=config.MODEL.DECODER.IN_CHANNELS,
            feature_key=config.MODEL.DECODER.FEATURE_KEY,
            low_level_channels=config.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_CHANNELS,
            low_level_key=config.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_KEY,
            low_level_channels_project=config.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_CHANNELS_PROJECT,
            decoder_channels=config.MODEL.DECODER.DECODER_CHANNELS,
            atrous_rates=config.MODEL.DECODER.ATROUS_RATES,
            num_classes=config.DATASET.NUM_CLASSES,
            has_instance=config.MODEL.PANOPTIC_DEEPLAB.INSTANCE.ENABLE,
            instance_low_level_channels_project=config.MODEL.PANOPTIC_DEEPLAB.INSTANCE.LOW_LEVEL_CHANNELS_PROJECT,
            instance_decoder_channels=config.MODEL.PANOPTIC_DEEPLAB.INSTANCE.DECODER_CHANNELS,
            instance_head_channels=config.MODEL.PANOPTIC_DEEPLAB.INSTANCE.HEAD_CHANNELS,
            instance_aspp_channels=config.MODEL.PANOPTIC_DEEPLAB.INSTANCE.ASPP_CHANNELS,
            instance_num_classes=config.MODEL.PANOPTIC_DEEPLAB.INSTANCE.NUM_CLASSES,
            instance_class_key=config.MODEL.PANOPTIC_DEEPLAB.INSTANCE.CLASS_KEY,
            semantic_loss=build_loss_from_cfg(config.LOSS.SEMANTIC),
            semantic_loss_weight=config.LOSS.SEMANTIC.WEIGHT,
            center_loss=build_loss_from_cfg(config.LOSS.CENTER),
            center_loss_weight=config.LOSS.CENTER.WEIGHT,
            offset_loss=build_loss_from_cfg(config.LOSS.OFFSET),
            offset_loss_weight=config.LOSS.OFFSET.WEIGHT,
        ),
    }

    # todo: replace the ResNet with NSA-architecture.
    if config.MODEL.BACKBONE.META == 'resnet':
        backbone = resnet.__dict__[config.MODEL.BACKBONE.NAME](
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
            replace_stride_with_dilation=model_cfg[config.MODEL.META_ARCHITECTURE]['replace_stride_with_dilation']
        )
    elif config.MODEL.BACKBONE.META == 'mobilenet_v2':
        backbone = mobilenet.__dict__[config.MODEL.BACKBONE.NAME](
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
        )
    elif config.MODEL.BACKBONE.META == 'mnasnet':
        backbone = mnasnet.__dict__[config.MODEL.BACKBONE.NAME](
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
        )
    elif config.MODEL.BACKBONE.META == 'nas':
        lasts = []
        for idx, arch_idx in enumerate(config.MODEL.NAS.ARCHI):
            if config.MODEL.NAS.LOAD_EPOCH == "last":
                state = torch.load(os.path.join(config.MODEL.NAS.LOAD_PATH, "arch_%d.pt" % arch_idx))
            else:
                state = torch.load(os.path.join(config.MODEL.NAS.LOAD_PATH, "arch_%d_%d.pt" % (arch_idx, int(config.load_epoch))))

            # TODO: revise the parameters here. according to FasterSeg.
            backbone = fasterseg.Network_Multi_Path_Infer([state["alpha_%d_0" % arch_idx].detach(), state["alpha_%d_1" % arch_idx].detach(),
                                                               state["alpha_%d_2" % arch_idx].detach()],
                                                              [None, state["beta_%d_1" % arch_idx].detach(), state["beta_%d_2" % arch_idx].detach()],
                                                              [state["ratio_%d_0" % arch_idx].detach(), state["ratio_%d_1" % arch_idx].detach(),
                     state["ratio_%d_2" % arch_idx].detach()],
                                                              num_classes=config.MODEL.NAS.NUM_CLASSES, layers=config.MODEL.NAS.LAYERS, Fch=config.MODEL.NAS.FCH,
                                                              width_mult_list=config.MODEL.NAS.WIDTH_MULT_LIST, stem_head_width=config.MODEL.NAS.STEM_HEAD_WIDTH[idx],
                                                              ignore_skip=arch_idx==0)
            mIoU02 = state["mIoU02"];
            latency02 = state["latency02"];
            obj02 = objective_acc_lat(mIoU02, latency02)
            mIoU12 = state["mIoU12"];
            latency12 = state["latency12"];
            obj12 = objective_acc_lat(mIoU12, latency12)
            if obj02 > obj12:
                last = [2, 0]
            else:
                last = [2, 1]
            lasts.append(last)
            backbone.build_structure(last)


            if arch_idx == 0 and len(config.MODEL.NAS.ARCHI) > 1:
                partial = torch.load(os.path.join(config.MODEL.NAS.TEACHER_PATH, "weights%d.pt" % arch_idx))
                state = backbone.state_dict()
                pretrained_dict = {k: v for k, v in partial.items() if k in state}
                state.update(pretrained_dict)
                backbone.load_state_dict(state)



    else:
        raise ValueError('Unknown meta backbone {}, please first implement it.'.format(config.MODEL.BACKBONE.META))

    model = model_map[config.MODEL.META_ARCHITECTURE](
        backbone,
        **model_cfg[config.MODEL.META_ARCHITECTURE]
    )
    # set batchnorm momentum
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = config.MODEL.BN_MOMENTUM
    return model


def build_loss_from_cfg(config):
    """Builds loss function with specific configuration.
    Args:
        config: the configuration.

    Returns:
        A nn.Module loss.
    """
    if config.NAME == 'cross_entropy':
        # return CrossEntropyLoss(ignore_index=config.IGNORE, reduction='mean')
        return RegularCE(ignore_label=config.IGNORE)
    elif config.NAME == 'ohem':
        return OhemCE(ignore_label=config.IGNORE, threshold=config.THRESHOLD, min_kept=config.MIN_KEPT)
    elif config.NAME == 'hard_pixel_mining':
        return DeepLabCE(ignore_label=config.IGNORE, top_k_percent_pixels=config.TOP_K_PERCENT)
    elif config.NAME == 'mse':
        return MSELoss(reduction=config.REDUCTION)
    elif config.NAME == 'l1':
        return L1Loss(reduction=config.REDUCTION)
    else:
        raise ValueError('Unknown loss type: {}'.format(config.NAME))
