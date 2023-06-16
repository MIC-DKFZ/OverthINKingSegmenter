from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
        return image


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image


class CT16Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float16):
        super().__init__(use_mask_for_norm, intensityproperties, target_dtype)

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = int(self.intensityproperties['percentile_00_5'])
        upper_bound = int(self.intensityproperties['percentile_99_5'])
        image = np.clip(image, lower_bound, upper_bound)
        image = image.astype(self.target_dtype)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image


class CT16NormalizationSlice(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float16, size: int=32, position="middle"):
        super().__init__(use_mask_for_norm, intensityproperties, target_dtype)
        self.size = size
        self.position = position

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        means = np.mean(image, (1,2))
        argmax = np.argmax(means)
        argmin = np.argmin(means)
        if self.position == "middle":
            argmean = (argmin+argmax)//2
            if argmean < self.size//2: argmean = self.size//2
            if argmean > image.shape[0]-self.size//2: argmean = image.shape[0]-self.size//2
            image = image[argmean-self.size//2:argmean+self.size//2]
        elif self.position == "first":
            if argmin < self.size: argmin = self.size
            image = image[argmin-self.size+1:argmin+1]
        else:
            raise NotImplementedError
        mean_intensity = np.mean(image[:, seg[0]>=0])
        std_intensity = np.std(image[:, seg[0]>=0])
        lower_bound = int(np.percentile(image[:, seg[0]>=0], 0.5))
        upper_bound = int(np.percentile(image[:, seg[0]>=0], 99.5))
        image = np.clip(image, lower_bound, upper_bound)
        image = image.astype(self.target_dtype)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        image[:, seg[0]==-1] = 0
        return image


class CT16NormalizationSliceStandardize(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        means = np.mean(image, (1,2))
        argmax = np.argmax(means)
        argmin = np.argmin(means)
        argmean = (argmin+argmax)//2
        if argmean < 16: argmean = 16
        if argmean > image.shape[0]-16: argmean = image.shape[0]-16
        image = image[argmean-16:argmean+16]
        image = np.split(image, image.shape[0])
        max_uint16 = np.iinfo(np.uint16).max
        for x in range(len(image)):
            pass
            slc = image[x].astype(np.float32)
            slc = (slc/max_uint16)*2-1
            image[x] = slc.astype(np.float16)
        return np.vstack(image)


class CT16NormalizationInd(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float16):
        super().__init__(use_mask_for_norm, intensityproperties, target_dtype)

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        points_per_slice = int(0.1*np.sum(seg[0]>=0))
        img_values = [np.random.choice(image[x, seg[0]>=0], points_per_slice) for x in range(image.shape[0])]
        mean_intensity = np.mean(img_values)
        std_intensity = np.std(img_values)
        lower_bound = int(np.percentile(img_values, 0.5))
        upper_bound = int(np.percentile(img_values, 99.5))
        image = np.clip(image, lower_bound, upper_bound)
        image = image.astype(self.target_dtype)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        image[:, seg[0]==-1] = 0
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = image - image.min()
        image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype)
        image = image / 255.
        return image

