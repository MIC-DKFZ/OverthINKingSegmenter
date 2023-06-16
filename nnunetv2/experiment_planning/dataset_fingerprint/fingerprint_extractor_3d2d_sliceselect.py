import os
from typing import List, Type, Union

import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero_3D2D
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_folder_identifiers_from_splitted_dataset_folder, \
    create_folder_lists_from_splitted_dataset_folder


class DatasetFingerprintExtractor3D2DSliceselect(object):
    def __init__(self, dataset_name_or_id: Union[str, int], num_processes: int = 1, verbose: bool = False):
        """
        extracts the dataset fingerprint used for experiment planning. The dataset fingerprint will be saved as a
        json file in the input_folder

        Philosophy here is to do only what we really need. Don't store stuff that we can easily read from somewhere
        else. Don't compute stuff we don't need (except for intensity_statistics_per_channel)
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.verbose = verbose

        self.dataset_name = dataset_name
        self.input_folder = join(nnUNet_raw, dataset_name)
        self.num_processes = num_processes
        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))

    @staticmethod
    def analyze_case(image_files: List[str], segmentation_file: str, reader_writer_class: Type[BaseReaderWriter]):
        rw = reader_writer_class()
        images, properties_images = rw.read_images(image_files)
        segmentation, properties_seg = rw.read_seg(segmentation_file)

        # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
        # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
        # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
        # way. This is only possible because we are now using our new input/output interface.
        data_cropped, seg_cropped, bbox = crop_to_nonzero_3D2D(images, segmentation)

        spacing = properties_images['spacing']

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        return shape_after_crop, spacing, relative_size_after_cropping

    def run(self, overwrite_existing: bool = False) -> dict:
        # we do not save the properties file in self.input_folder because that folder might be read-only. We can only
        # reliably write in nnUNet_preprocessed and nnUNet_results, so nnUNet_preprocessed it is
        preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')

        if not isfile(properties_file) or overwrite_existing:
            file_ending = self.dataset_json['file_ending']
            training_identifiers = get_folder_identifiers_from_splitted_dataset_folder(join(self.input_folder, 'imagesTr'))
            reader_writer_class = determine_reader_writer_from_dataset_json(self.dataset_json,
                                                                            join(self.input_folder, 'imagesTr',
                                                                                 training_identifiers[
                                                                                     0] + '_0000' + file_ending))

            training_images_per_case = create_folder_lists_from_splitted_dataset_folder(join(self.input_folder, 'imagesTr'))
            training_labels_per_case = [join(self.input_folder, 'labelsTr', i + file_ending) for i in
                                        training_identifiers]

            results = ptqdm(DatasetFingerprintExtractor3D2DSliceselect.analyze_case,
                            (training_images_per_case, training_labels_per_case),
                            processes=self.num_processes, zipped=True, reader_writer_class=reader_writer_class,
                            disable=self.verbose)

            shapes_after_crop = [r[0] for r in results]
            spacings = [r[1] for r in results]

            # we drop this so that the json file is somewhat human readable
            # foreground_intensity_stats_by_case_and_modality = [r[3] for r in results]
            median_relative_size_after_cropping = np.median([r[2] for r in results], 0)

            fingerprint = {
                    "spacings": spacings,
                    "shapes_after_crop": shapes_after_crop,
                    "median_relative_size_after_cropping": median_relative_size_after_cropping
                }

            try:
                save_json(fingerprint, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)
        return fingerprint


if __name__ == '__main__':
    dfe = DatasetFingerprintExtractor3D2DSliceselect(763, 1, True)
    dfe.run(overwrite_existing=False)
