import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from nnunetv2.architecture.ResEnc_UNet_encoder3D import ResEncUNetEncoder3D
from nnunetv2.architecture.UNet_decoder2D import UNetDecoder2D
from nnunetv2.architecture.utils import compute_skip_sizes, SqEx_Block


class ResEncUNet3DSqEx2D(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 patch_size: Union[List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 dropout_op_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResEncUNetEncoder3D(input_channels, n_stages, features_per_stage, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, dropout_op_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        skip_sizes = compute_skip_sizes(patch_size, strides)
        self.decoder = UNetDecoder2D(features_per_stage, n_stages, kernel_sizes, strides, n_conv_per_stage_decoder, 
                                     num_classes, deep_supervision, conv_bias, False, dropout_op_kwargs=dropout_op_kwargs)
        self.SqExBlocks = nn.ModuleList(SqEx_Block(skip_size[0], 2*skip_size[0]) for skip_size in skip_sizes)

    def forward(self, x):
        skips = self.encoder(x)
        skips = [sqex_block(skip) for sqex_block, skip in zip(self.SqExBlocks, skips)]
        skips = [F.adaptive_avg_pool3d(skip, (1, *skip.shape[-2:])).squeeze_(2) for skip in skips]
        ret = self.decoder(skips)
        if isinstance(ret, (list,tuple)):
            return [r.unsqueeze_(2) for r in ret]
        return ret.unsqueeze_(2)