from time import time

import argparse
import imageio
import numpy as np
from pathlib import Path
import tifffile
import torch
from torch import nn

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


def normalize(image, fpe):
    mean_intensity = fpe["mean"]
    std_intensity = fpe["std"]
    lower_bound = int(fpe["lower_bound"])
    upper_bound = int(fpe["upper_bound"])
    image = np.clip(image, lower_bound, upper_bound)
    image = image.astype(np.float16)
    image = (image - mean_intensity) / max(std_intensity, 1e-8)
    return image


def get_stats(data, mask):
    points_per_slice = int(0.1 * np.sum(mask))
    means = []
    reps = []
    for x in range(len(data)):
        data_slice = np.copy(data[x])
        reps.append(np.random.choice(data_slice[mask], points_per_slice))
        means.append(np.mean(reps[-1]))
    argmax = np.argmax(means)
    argmin = np.argmin(means)
    argmean = (argmax + argmin) // 2
    if argmean < 16:
        argmean = 16
    if argmean > len(data) - 16:
        argmean = len(data) - 16
    x_slice = (argmean - 16, argmean + 16)
    reps = reps[x_slice[0] : x_slice[1]]
    statistics = {
        "lower_bound": np.percentile(reps, 0.5),
        "upper_bound": np.percentile(reps, 99.5),
        "mean": np.mean(reps),
        "std": np.std(reps),
    }
    return x_slice, statistics


def predict_sliding_window_return_logits3D2D(
    network: nn.Module,
    data: Union[np.ndarray, torch.Tensor],
    num_segmentation_heads: int,
    tile_size: Union[Tuple[int, ...], List[int]],
    fingerprint: dict,
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

            if use_gaussian:
                gaussian = (
                    compute_gaussian(
                        tuple(tile_size[1:]),
                        sigma_scale=1.0 / 8,
                        value_scaling_factor=1000,
                        device=device,
                    )
                    if precomputed_gaussian is None
                    else precomputed_gaussian
                )

            slicers = list(
                get_sliding_window_generator(
                    (len(data), *data[0].shape),
                    tile_size,
                    tile_step_size,
                    verbose=verbose,
                )
            )

            # preallocate results and num_predictions. Move everything to the correct device
            try:
                predicted_logits = torch.zeros(
                    (num_segmentation_heads, 1, *data[0].shape),
                    dtype=torch.half,
                    device=results_device,
                )
                n_predictions = torch.zeros(
                    (1, *data[0].shape), dtype=torch.half, device=results_device
                )
                gaussian = gaussian.to(results_device)
            except RuntimeError:
                # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                results_device = torch.device("cpu")
                predicted_logits = torch.zeros(
                    (num_segmentation_heads, 1, *data[0].shape),
                    dtype=torch.half,
                    device=results_device,
                )
                n_predictions = torch.zeros(
                    (1, *data[0].shape), dtype=torch.half, device=results_device
                )
                gaussian = gaussian.to(results_device)
            finally:
                empty_cache(device)

            for sl in slicers:
                workon = np.stack([d[sl[2:]] for d in data])[None][None]
                workon = normalize(workon, fingerprint)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    workon = torch.from_numpy(workon)
                workon = workon.to(device, non_blocking=False)

                prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[
                    0
                ].to(results_device)
                prediction += torch.rot90(
                    maybe_mirror_and_predict(
                        network, torch.rot90(workon, 1, [-2, -1]), mirror_axes
                    )[0].to(results_device),
                    -1,
                    [-2, -1],
                )

                prediction /= 2

                predicted_logits[sl] += (
                    prediction * gaussian if use_gaussian else prediction
                )
                n_predictions[sl[1:]] += gaussian if use_gaussian else 1

            predicted_logits /= n_predictions
    empty_cache(device)
    return predicted_logits.mean(1)  # Mean over x dim to return 2D logits


def predict_from_memmap_list(
    trainer, parameter_list, folder, output_folder, mirror_axes, patch_x
):
    folder = Path(folder)
    output_folder = Path(output_folder)
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()

    num_seg_heads = trainer.label_manager.num_segmentation_heads

    files = sorted([f for f in folder.iterdir() if f.match("*.tif")])
    data = [tifffile.memmap(f, mode="r") for f in files]
    mask = imageio.imread_v2(folder / "../mask.png")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask > 0
    x_sl, fingerprint = get_stats(data, mask)
    data = data[x_sl[0] : x_sl[1]]
    print(f"data shape is {(len(data), *data[0].shape)}")

    output_filename = Path(output_folder) / f"{folder.parent.name}.npy"
    output_filename_seg = Path(output_folder) / f"{folder.parent.name}_seg.npy"

    prediction = None
    for i, parameters in enumerate(parameter_list):
        print(f"using parameters {i}")
        trainer.network.load_state_dict(parameters)
        try:
            prediction_logits = predict_sliding_window_return_logits3D2D(
                trainer.network,
                data,
                num_seg_heads,
                tile_size=trainer.configuration_manager.patch_size,
                mirror_axes=mirror_axes,
                tile_step_size=0.5,
                fingerprint=fingerprint,
                use_gaussian=True,
                precomputed_gaussian=None,
                perform_everything_on_gpu=True,
                verbose=False,
                device=trainer.device,
            ).cpu()
        except RuntimeError:
            prediction_logits = predict_sliding_window_return_logits3D2D(
                trainer.network,
                data,
                num_seg_heads,
                tile_size=trainer.configuration_manager.patch_size,
                mirror_axes=mirror_axes,
                tile_step_size=0.5,
                fingerprint=fingerprint,
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

    np.save(output_filename_seg, prediction)
    np.save(output_filename, predicted_mask)

    trainer.set_deep_supervision_enabled(True)
    compute_gaussian.cache_clear()


def convert_to_csv(output_folder):
    output_folder = Path(output_folder)
    npy_files = sorted(
        [
            f
            for f in output_folder.iterdir()
            if f.match("*.npy") and not f.match("*_seg.npy")
        ]
    )
    inklabels_rle = dict()
    for npy_file in npy_files:
        mask = np.load(npy_file)
        starts_ix, lengths = rle(mask)
        inklabels_rle[npy_file.name[:-4]] = " ".join(
            map(str, sum(zip(starts_ix, lengths), ()))
        )

    csv_path = Path(output_folder) / "submission.csv"
    with open(csv_path, "w") as output_file:
        output_file.write("Id,Predicted\n")
        for k in inklabels_rle.keys():
            output_file.write(f"{k},{inklabels_rle[k]}\n")


def predict_overthinking_segmenter_32ind_tta(
    input_folder, output_folder, trainer, plans_file, dataset_json_file, checkpoint_path
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    test_cases = sorted([d for d in input_folder.iterdir() if d.is_dir()])
    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)
    trainer = trainer(plans, "3d_fullres", 0, dataset_json)
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
    patch_x = plans["configurations"]["3d_fullres"]["patch_size"][0]

    for test_case in test_cases:
        print(f"predicting case {test_case.name}")
        img_folder = test_case / "surface_volume"
        if not img_folder.is_dir():
            continue
        predict_from_memmap_list(
            trainer, parameters, img_folder, output_folder, mirror_axes, patch_x
        )

    convert_to_csv(output_folder)


if __name__ == "__main__":
    input_folder = Path("vesuvius/unwrapped_scrolls/data")
    output_folder = Path("vesuvius/unwrapped_scrolls/predictions")
    conf_dir = Path(
        "vesuvius/submissions/764-nnUNetTrainer3DSqEx2D-ndbd2-bd4-large4conv"
    )
    plans_file = conf_dir / "plans.json"
    dataset_json_file = conf_dir / "dataset.json"
    checkpoint_path = conf_dir

    trainer = "nnUNetTrainer3DSqEx2D"

    nnUNetTrainer = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer,
        "nnunetv2.training.nnUNetTrainer",
    )

    t_start = time()

    predict_overthinking_segmenter_32ind_tta(
        input_folder,
        output_folder,
        nnUNetTrainer,
        plans_file,
        dataset_json_file,
        checkpoint_path,
    )

    print(f"prediction took {time()-t_start} seconds")
