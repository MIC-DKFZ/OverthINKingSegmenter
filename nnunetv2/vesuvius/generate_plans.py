import argparse
from copy import deepcopy
import os
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json

nnUNet_preprocessed = Path(os.environ["nnUNet_preprocessed"])


def generate_plans(dataset):
    basic_plans = load_json(nnUNet_preprocessed/dataset/"nnUNetPlans.json")
    save_json(basic_plans, nnUNet_preprocessed/dataset/"nnUNetPlans_old.json", sort_keys=False) # default which we don't want
    basic_plans["configurations"]["3d_fullres"]["conv_kernel_sizes"] = \
        [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    save_json(basic_plans, nnUNet_preprocessed/dataset/"nnUNetPlans.json", sort_keys=False)
    submission_plans_bs2 = deepcopy(basic_plans)
    submission_plans_bs4 = deepcopy(basic_plans)
    bs2_name = "nnUNetPlans_large_4conv"
    bs4_name = "nnUNetPlans_bs4_large_4conv"
    submission_plans_bs2["plans_name"] = bs2_name
    submission_plans_bs4["plans_name"] = bs4_name
    submission_plans_bs2["configurations"]["3d_fullres"]["patch_size"] = [32, 512, 512]
    submission_plans_bs2["configurations"]["3d_fullres"]["n_conv_per_stage_encoder"] = [4, 4, 4, 4, 4, 4, 4]
    submission_plans_bs4["configurations"]["3d_fullres"]["batch_size"] = 4
    submission_plans_bs4["configurations"]["3d_fullres"]["patch_size"] = [32, 512, 512]
    submission_plans_bs4["configurations"]["3d_fullres"]["n_conv_per_stage_encoder"] = [4, 4, 4, 4, 4, 4, 4]
    save_json(submission_plans_bs2, nnUNet_preprocessed/dataset/f"{bs2_name}.json", sort_keys=False)
    save_json(submission_plans_bs4, nnUNet_preprocessed/dataset/f"{bs4_name}.json", sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Dataset801_vesuvius_split")
    args = parser.parse_args()
    generate_plans(args.dataset)
