import argparse
import numpy as np
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def find_data_location(data_path, case):
    try:
        seg = np.load(data_path/f"{case}_seg.npy")
    except FileNotFoundError:
        with np.load(data_path/f"{case}.npz") as f:
            seg = f["seg"]
    properties = load_pickle(data_path/f"{case}.pkl")
    properties["data_locations"] = np.argwhere(seg>=0)
    save_pickle(properties, data_path/f"{case}.pkl")


def process_dataset(dataset):
    data_path = Path(nnUNet_preprocessed)/dataset/"nnUNetPlans_3d_fullres"
    cases = sorted([c.name[:-4] for c in data_path.iterdir() if c.match("*.npz")])
    for case in cases:
        find_data_location(data_path, case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="dataset to process")
    args = parser.parse_args()

    #dataset = maybe_convert_to_dataset_name(args.dataset)
    dataset = maybe_convert_to_dataset_name("762")
    process_dataset(dataset)