import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer3DSqEx2D import nnUNetTrainer3DSqEx2D, nnUNetTrainer3DSqEx2DLongTraining
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
import numpy as np
from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs
from typing import Union, Tuple, List

class nnUNetTrainer3DSqEx2D_wd(nnUNetTrainer3DSqEx2D):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_decay = 1e-4


class nnUNetTrainer3DSqEx2DLongTraining_wd(nnUNetTrainer3DSqEx2DLongTraining):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.weight_decay = 1e-4

class nnUNetTrainer3DSqEx2D_wd_normal_dice(nnUNetTrainer3DSqEx2D_wd):
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

class nnUNetTrainer3DSqEx2D_wd_5_epochs(nnUNetTrainer3DSqEx2D_wd):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5