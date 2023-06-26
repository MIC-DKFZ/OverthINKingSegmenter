import argparse
import numpy as np
import os
from pathlib import Path
import random
import shutil
import tifffile

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json, load_pickle, save_pickle

nnUNet_preprocessed = Path(os.environ["nnUNet_preprocessed"])


def find_split(seg):
    seg = seg.reshape(seg.shape[-2:])
    total = np.sum(seg>=0)
    min_x = np.min(np.argwhere(seg>=0)[:,0])
    max_x = np.max(np.argwhere(seg>=0)[:,0])
    xy_split = dict()
    x_split = {0: [min_x, None], 1: [None, None], 2: [None, None], 3: [None, None], 4: [None, max_x]}
    current = min_x
    for i in range(4):
        for x in range(current+1, seg.shape[0]):
            if np.sum(seg[current:x]>=0) >= total//5:
                current = x_split[i][1] = x_split[i+1][0] = x
                break
    for i in range(5):
        x_total = np.sum(seg[x_split[i][0]:x_split[i][1]]>=0)
        min_y = np.min(np.argwhere(seg[x_split[i][0]:x_split[i][1]]>=0)[:,1])
        max_y = np.max(np.argwhere(seg[x_split[i][0]:x_split[i][1]]>=0)[:,1])
        y_split = {0: [min_y, None], 1: [None, None], 2: [None, None], 3: [None, None], 4: [None, max_y]}
        current = min_y
        for j in range(4):
            for y in range(current+1, seg.shape[1]):
                if np.sum(seg[x_split[i][0]:x_split[i][1], current:y]>=0) >= x_total//5:
                    current = y_split[j][1] = y_split[j+1][0] = y
                    break
        for j in range(5):
            xy_split[(i, j)] = (x_split[i], y_split[j])
    return xy_split


def split_and_save(old_dataset, new_dataset, case):
    with np.load(old_dataset/"nnUNetPlans_3d_fullres"/f"{case}.npz") as f:
        data = f["data"]
        seg = f["seg"]
    meta = load_pickle(old_dataset/"nnUNetPlans_3d_fullres"/f"{case}.pkl")
    gt = tifffile.imread(old_dataset/"gt_segmentations"/f"{case}.tif")
    bbox = meta["bbox_used_for_cropping"]
    gt = gt[slice(*bbox[1]), slice(*bbox[2])]
    xy_split = find_split(seg)
    for i, k in enumerate(xy_split.keys(), 1): # such a highly optimized for loop ;)
        name = "{:s}_{:02d}".format(case, i)
        x_slice = slice(*xy_split[k][0])
        y_slice = slice(*xy_split[k][1])
        data_slice = data[:, :, x_slice, y_slice]
        seg_slice = seg[:, :, x_slice, y_slice]
        np.savez(new_dataset/"nnUNetPlans_3d_fullres"/f"{name}.npz", data=data_slice, seg=seg_slice)
        gt_slice = gt[x_slice, y_slice]
        tifffile.imwrite(new_dataset/"gt_segmentations"/f"{name}.tif", gt_slice)
        meta_slice = meta.copy()
        meta_slice["shape_before_cropping"] = data_slice.shape[-3:]
        meta_slice["bbox_used_for_cropping"] = [[0, 1], [0, data_slice.shape[-2]], [0, data_slice.shape[-1]]]
        meta_slice["shape_after_cropping_and_before_resampling"] = data_slice.shape[-3:]
        meta_slice["class_locations"] = {1: np.argwhere(seg_slice==1)}
        meta_slice["data_locations"] = np.argwhere(seg_slice>=0)
        save_pickle(meta_slice, new_dataset/"nnUNetPlans_3d_fullres"/f"{name}.pkl")


def create_new_dataset(old_task_name, new_task_name):
    # create new task with data split into 25 parts each
    old_task = nnUNet_preprocessed/old_task_name
    new_task = nnUNet_preprocessed/new_task_name
    new_task.mkdir(exist_ok=True)
    (new_task/"gt_segmentations").mkdir(exist_ok=True)
    (new_task/"nnUNetPlans_3d_fullres").mkdir(exist_ok=True)
    dataset = load_json(old_task/"dataset.json")
    dataset["name"] = new_task_name
    save_json(dataset, new_task/"dataset.json", sort_keys=False)
    shutil.copy(old_task/"dataset_fingerprint.json", new_task/"dataset_fingerprint.json")
    plans = load_json(old_task/"nnUNetPlans.json")
    plans["dataset_name"] = new_task_name
    save_json(plans, new_task/"nnUNetPlans.json", sort_keys=False)
    cases = sorted([c.name[:-4] for c in (old_task/"gt_segmentations").iterdir() if c.match("*.tif")])
    for case in cases:
        split_and_save(old_task, new_task, case)

    # create seeded split file for 5 fold cv training. every split contains 5 parts from each scroll
    folds = [[] for _ in range(5)]
    for case in cases:
        new_case = ["{:s}_{:02d}".format(case, i) for i in range(1, 26)]
        random.Random(42).shuffle(new_case)
        for i in range(5):
            folds[i].extend(new_case[i*5:(i+1)*5])
    splits = []
    splits.append({"train": sorted(folds[0]+folds[1]+folds[2]+folds[3]), "val": sorted(folds[4])})
    splits.append({"train": sorted(folds[0]+folds[1]+folds[2]+folds[4]), "val": sorted(folds[3])})
    splits.append({"train": sorted(folds[0]+folds[1]+folds[3]+folds[4]), "val": sorted(folds[2])})
    splits.append({"train": sorted(folds[0]+folds[2]+folds[3]+folds[4]), "val": sorted(folds[1])})
    splits.append({"train": sorted(folds[1]+folds[2]+folds[3]+folds[4]), "val": sorted(folds[0])})

    save_json(splits, new_task/"splits_final.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_dataset", default="Dataset800_vesuvius")
    parser.add_argument("--new_dataset", default="Dataset801_vesuvius_split")
    args = parser.parse_args()
    create_new_dataset(args.old_dataset, args.new_dataset)
