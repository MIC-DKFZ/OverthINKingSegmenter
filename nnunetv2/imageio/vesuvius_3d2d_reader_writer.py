#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os.path
from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import tifffile
from batchgenerators.utilities.file_and_folder_operations import subfiles, isdir, save_json, join


class Vesuvius3D2DIO(BaseReaderWriter):
    """
    reads and writes 3D tif(f) images with corresponding 2D segmentations. Uses tifffile package. Ignores metadata (for now)!
    expects a folder with a sequence of 2D tif(f) images which are stacked to a 3D volume

    Uses isotropic spacing of (1, 1, 1)!
    """

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        assert isinstance(image_fnames, (list,tuple,np.ndarray)), f"invalid type for image_fnames {type(image_fnames)}"
        images = []
        for f in image_fnames:
            assert isdir(f), ("Please give valid folder name! Folder: %s" % f)
            image = np.stack([tifffile.imread(x) for x in subfiles(f, suffix="tif")])
            assert image.ndim == 3, ("Only 2D images are supported! Folder: %s" % f)
            images.append(image[None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        images = np.vstack(images)
        return images, {'spacing': (1, 1, 1)}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict = None) -> None:
        assert seg.ndim == 3 or seg.shape[0] == 1, f"Invalid shape {seg.shape} for segmentation {output_fname}"
        if seg.ndim == 2:
            seg = np.expand_dims(seg, 0)
        tifffile.imwrite(output_fname, data=seg.astype(np.uint8), compression='zlib')
        if properties:
            assert isinstance(properties, dict), f"please give properties as dict, {type(properties)} is not supported"
            file = os.path.basename(output_fname)
            out_dir = os.path.dirname(output_fname)
            ending = file.split('.')[-1]
            save_json({'spacing': properties.get('spacing', (1, 1, 1))}, join(out_dir, file[:-(len(ending) + 1)] + '.json'))

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        seg = tifffile.imread(seg_fname)
        if len(seg.shape) != 3:
            seg = np.expand_dims(seg, 0)
        seg = seg[None]

        return seg.astype(np.int8), {'spacing': (1, 1, 1)}