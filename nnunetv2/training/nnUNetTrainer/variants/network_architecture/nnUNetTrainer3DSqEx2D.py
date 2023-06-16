import torch

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import nn
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He

from nnunetv2.architecture.UNet3DSqEx2D import UNet3DSqEx2D
from nnunetv2.architecture.ResEncUNet3DSqEx2D import ResEncUNet3DSqEx2D
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_base3d2d import nnUNetTrainerBase3D2D


class nnUNetTrainer3DSqEx2D(nnUNetTrainerBase3D2D):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True,
                                   dropout_prob: int = 0) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dropout_op_kwargs = {"p": dropout_prob, "inplace": True} if dropout_prob > 0 else None

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = configuration_manager.UNet_class_name
        mapping = {
            'PlainConvUNet': UNet3DSqEx2D,
            'ResidualEncoderUNet': ResEncUNet3DSqEx2D
        }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                "dropout_op_kwargs": dropout_op_kwargs,
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                "dropout_op_kwargs": dropout_op_kwargs,
            },
        }

        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                'into either this ' \
                                                                'function (get_network_from_plans) or ' \
                                                                'the init of your nnUNetModule to accomodate that.'
        network_class = mapping[segmentation_network_class_name]

        if network_class == ResEncUNet3DSqEx2D:
            conv_or_blocks_per_stage = {
            'n_blocks_per_stage': [n//2 for n in configuration_manager.n_conv_per_stage_encoder],
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
            }

        else:
            conv_or_blocks_per_stage = {
            'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
            }

        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            patch_size=configuration_manager.patch_size,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )

        model.apply(InitWeights_He(1e-2))
        if network_class == ResEncUNet3DSqEx2D:
            model.apply(init_last_bn_before_add_to_0)
        return model


class nnUNetTrainer3DSqEx2DLongTraining(nnUNetTrainer3DSqEx2D):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000


class nnUNetTrainer3DSqEx2D_normal_dice(nnUNetTrainer3DSqEx2D):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.beta = 1