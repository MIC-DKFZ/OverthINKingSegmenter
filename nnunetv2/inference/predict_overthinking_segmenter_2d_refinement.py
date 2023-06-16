import imageio
import numpy as np
from pathlib import Path
import torch
from torch import nn
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from typing import Union, Tuple, List
import warnings

from batchgenerators.utilities.file_and_folder_operations import List, load_json, join
import nnunetv2
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian,
    get_sliding_window_generator,
    maybe_mirror_and_predict,
)
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import nnunetv2
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache

softmax = nn.Softmax(dim=0)


def rle(img):
    flat_img = img.flatten()
    flat_img[-1] = 0
    flat_img[0] = 0
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    return starts_ix, lengths


def predict_sliding_window_return_logits(
    network: nn.Module,
    input_image: Union[np.ndarray, torch.Tensor],
    num_segmentation_heads: int,
    tile_size: Union[Tuple[int, ...], List[int]],
    mirror_axes: Tuple[int, ...] = None,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    precomputed_gaussian: torch.Tensor = None,
    perform_everything_on_gpu: bool = True,
    verbose: bool = True,
    device: torch.device = torch.device("cuda"),
) -> Union[np.ndarray, torch.Tensor]:
    if perform_everything_on_gpu:
        assert (
            device.type == "cuda"
        ), 'Can use perform_everything_on_gpu=True only when device="cuda"'

    network = network.to(device)
    network.eval()

    empty_cache(device)

    with torch.no_grad():
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(
            device.type, enabled=True
        ) if device.type == "cuda" else dummy_context():
            assert (
                len(input_image.shape) == 4
            ), "input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)"

            if not torch.cuda.is_available():
                if perform_everything_on_gpu:
                    print(
                        'WARNING! "perform_everything_on_gpu" was True but cuda is not available! Set it to False...'
                    )
                perform_everything_on_gpu = False

            results_device = (
                device if perform_everything_on_gpu else torch.device("cpu")
            )

            if verbose:
                print("step_size:", tile_step_size)
            if verbose:
                print("mirror_axes:", mirror_axes)

            if not isinstance(input_image, torch.Tensor):
                # pytorch will warn about the numpy array not being writable. This doesnt matter though because we
                # just want to read it. Suppress the warning in order to not confuse users...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_image = torch.from_numpy(input_image)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(
                input_image, tile_size, "constant", {"value": 0}, True, None
            )

            if use_gaussian:
                gaussian = (
                    compute_gaussian(
                        tuple(tile_size),
                        sigma_scale=1.0 / 8,
                        value_scaling_factor=1000,
                        device=device,
                    )
                    if precomputed_gaussian is None
                    else precomputed_gaussian
                )

            slicers = get_sliding_window_generator(
                data.shape[1:], tile_size, tile_step_size, verbose=verbose
            )

            # preallocate results and num_predictions. Move everything to the correct device
            try:
                predicted_logits = torch.zeros(
                    (num_segmentation_heads, *data.shape[1:]),
                    dtype=torch.half,
                    device=results_device,
                )
                n_predictions = torch.zeros(
                    data.shape[1:], dtype=torch.half, device=results_device
                )
                gaussian = gaussian.to(results_device)
            except RuntimeError:
                # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                results_device = torch.device("cpu")
                predicted_logits = torch.zeros(
                    (num_segmentation_heads, *data.shape[1:]),
                    dtype=torch.half,
                    device=results_device,
                )
                n_predictions = torch.zeros(
                    data.shape[1:], dtype=torch.half, device=results_device
                )
                gaussian = gaussian.to(results_device)
            finally:
                empty_cache(device)

            for sl in slicers:
                workon = data[sl][None]
                workon = torch.nn.functional.interpolate(
                    workon, size=(1024, 1024), mode="bilinear"
                )
                workon = workon.to(device, non_blocking=False)

                prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[
                    0
                ].to(results_device)[None]

                # Convert 'Half' tensor to 'Float' tensor
                prediction = torch.nn.functional.interpolate(
                    prediction, size=(2048, 2048), mode="bilinear", align_corners=False
                )

                prediction = prediction.squeeze()

                predicted_logits[sl] += (
                    prediction * gaussian if use_gaussian else prediction
                )
                n_predictions[sl[1:]] += gaussian if use_gaussian else 1

            predicted_logits /= n_predictions
    empty_cache(device)
    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]


def predict_from_list(
    trainer,
    parameter_list,
    test_case,
    old_input_folder,
    output_folder,
    mirror_axes,
    patch_x,
):
    old_input_folder = Path(old_input_folder)
    output_folder = Path(output_folder)
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()

    num_seg_heads = trainer.label_manager.num_segmentation_heads

    data = np.load(test_case)[None][None]
    mask = imageio.imread_v2(old_input_folder / f"{test_case.name[:-12]}/mask.png")

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask > 0

    print(f"data shape is {data[0].shape}")
    output_filename = Path(output_folder) / f"{test_case.name[:-12]}_refined.npy"

    prediction = None
    for i, parameters in enumerate(parameter_list):
        print(f"using parameters {i}")
        trainer.network.load_state_dict(parameters)
        try:
            prediction_logits = predict_sliding_window_return_logits(
                trainer.network,
                data,
                num_seg_heads,
                tile_size=trainer.configuration_manager.patch_size,
                mirror_axes=mirror_axes,
                tile_step_size=0.5,
                use_gaussian=True,
                precomputed_gaussian=None,
                perform_everything_on_gpu=True,
                verbose=False,
                device=trainer.device,
            ).cpu()
        except RuntimeError:
            prediction_logits = predict_sliding_window_return_logits(
                trainer.network,
                data,
                num_seg_heads,
                tile_size=trainer.configuration_manager.patch_size,
                mirror_axes=mirror_axes,
                tile_step_size=0.5,
                use_gaussian=True,
                precomputed_gaussian=None,
                perform_everything_on_gpu=False,
                verbose=False,
                device=trainer.device,
            ).cpu()
        if prediction is None:
            prediction = softmax(prediction_logits.float())
        else:
            prediction += softmax(prediction_logits.float())

    prediction = prediction.numpy()
    if len(parameter_list) > 1:
        prediction /= len(parameter_list)

    predicted_mask = np.argmax(prediction, 0)

    if predicted_mask.ndim == 3:
        predicted_mask[:, ~mask] = 0
    elif predicted_mask.ndim == 2:
        predicted_mask[~mask] = 0

    np.save(output_filename, predicted_mask)

    trainer.set_deep_supervision_enabled(True)
    compute_gaussian.cache_clear()


def convert_to_csv(output_folder):
    output_folder = Path(output_folder)
    npy_files = sorted([f for f in output_folder.iterdir() if f.match("*_refined.npy")])
    inklabels_rle = dict()
    for npy_file in npy_files:
        mask = np.load(npy_file)
        starts_ix, lengths = rle(mask)
        inklabels_rle[npy_file.name[:-12]] = " ".join(
            map(str, sum(zip(starts_ix, lengths), ()))
        )

    csv_path = Path(output_folder) / "submission.csv"
    with open(csv_path, "w") as output_file:
        output_file.write("Id,Predicted\n")
        for k in inklabels_rle.keys():
            output_file.write(f"{k},{inklabels_rle[k]}\n")


def predict_overthinking_segmenter_2d_refinement(
    old_input_folder,
    softmax_folder,
    output_folder,
    trainer,
    plans_file,
    dataset_json_file,
    checkpoint_path,
):
    old_input_folder = Path(old_input_folder)
    output_folder = Path(output_folder)
    softmax_folder = Path(softmax_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    test_cases = sorted(
        [f for f in softmax_folder.iterdir() if f.match("*_softmax.npy")]
    )

    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)
    trainer = trainer(plans, "2d", 0, dataset_json)
    trainer.initialize()
    if checkpoint_path.is_file():
        checkpoint_files = [checkpoint_path]
    else:
        checkpoint_files = sorted(
            [f for f in checkpoint_path.iterdir() if f.match("*.pth")]
        )
    parameters = []
    for i, checkpoint_file in enumerate(checkpoint_files):
        checkpoint = torch.load(str(checkpoint_file), map_location=torch.device("cpu"))
        if i == 0:
            mirror_axes = checkpoint["inference_allowed_mirroring_axes"]
        parameters.append(checkpoint["network_weights"])
    patch_x = plans["configurations"]["2d"]["patch_size"][0]

    for test_case in test_cases:
        print(f"predicting case {test_case}")
        predict_from_list(
            trainer,
            parameters,
            test_case,
            old_input_folder,
            output_folder,
            mirror_axes,
            patch_x,
        )
    convert_to_csv(output_folder)


if __name__ == "__main__":
    old_input_folder = Path("validation_splits/split_0")
    output_folder = Path(
        "nnUNet_results/Dataset764_overthinking_segmenter_split_sliceselect/nnUNetTrainer3DSqEx2D_wd__nnUNetPlans_bs4_large_4conv__3d_fullres/fold_0/refinement_final"
    )
    conf_dir = Path("refinement_final")
    softmax_folder = Path(
        "nnUNet_results/Dataset764_overthinking_segmenter_split_sliceselect/nnUNetTrainer3DSqEx2D_wd__nnUNetPlans_bs4_large_4conv__3d_fullres/fold_0/validation_best_padding_base"
    )

    plans_file = conf_dir / "plans.json"
    dataset_json_file = conf_dir / "dataset.json"
    checkpoint_path = conf_dir

    trainer = "nnUNetTrainer2DRefinement"

    nnUNetTrainer = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer,
        "nnunetv2.training.nnUNetTrainer",
    )

    predict_overthinking_segmenter_2d_refinement(
        old_input_folder,
        softmax_folder,
        output_folder,
        nnUNetTrainer,
        plans_file,
        dataset_json_file,
        checkpoint_path,
    )
