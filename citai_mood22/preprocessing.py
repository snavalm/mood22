
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import numpy as np

from monai.transforms import *
from monai.utils import ensure_tuple

from monai.data import (
    DataLoader,
    CacheDataset, Dataset,
    load_decathlon_datalist,
    load_decathlon_properties,
    decollate_batch,
    list_data_collate,
    NumpyReader, NibabelReader
)

import random
from monai.transforms.transform import MapTransform, Randomizable
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor, PathLike
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Union

from copy import deepcopy

from .preprocessing_utils import *

def get_transforms(keys=['image'], # Assumes at least image. If others are passed, only "Image" gets intensity transforms
                   spacing=None,
                   scale_source_intensity_values=None,
                   scale_source_intensity_percentile=None,
                   patch_size=None,
                   spatial_augmentations=False,
                   intesity_augmentations=False,
                   preserve_zero_background=True,  # If we apply intensity_augmentations, preserve background as 0
                   normalize=False,
                   normalize_nonzero=True,
                   normalize_subtrahend_divisor = None,
                   in_memory=False,
                   ):

    load_image = LoadImaged(keys=keys, image_only=True, ensure_channel_first=True, allow_missing_keys=True)
    load_image.register(RADChestCTReader()) # Add the custom RADChestCT reader to the Image Loader

    transforms = [
        load_image,
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
    ]

    if spacing is not None:
        if (patch_size is not None) and (not in_memory):   # If we don't keep the datase in memory, do patch before spacing
            pre_patch_size = [int(i * 1.25) for i in patch_size] if spatial_augmentations else patch_size
            transforms += [RandSpatialCropPreSpacingd(keys=keys, spacing=spacing, roi_size=pre_patch_size, allow_missing_keys=True)]

        transforms += [Spacingd(keys=keys, pixdim=spacing, padding_mode='zeros', align_corners=True, allow_missing_keys=True),
                       CastToTyped(keys=["image"],dtype=torch.float16) # Cast to float16 to save memory in cache
                       ]

    if scale_source_intensity_values is not None:
        transforms += [ScaleIntensityRanged(keys=["image"],
                                            a_min=scale_source_intensity_values[0],
                                            a_max=scale_source_intensity_values[1],
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=True, ), ]
    elif scale_source_intensity_percentile is not None:
        transforms += [ScaleIntensityRangePercentilesd(keys=["image"],
                                                       lower=scale_source_intensity_percentile[0],
                                                       upper=scale_source_intensity_percentile[1],
                                                       b_min=0.0,
                                                       b_max=1.0,
                                                       clip=True, ), ]

    if normalize and (normalize_subtrahend_divisor is None):
        transforms += [ComputeIntensityMeandStdd(keys=['image'],nonzero=normalize_nonzero)]

    # If we are going to apply spatial augmentations, patch before to avoid border effects
    if spatial_augmentations and (patch_size is not None):
        pre_patch_size = [int(i*1.25) for i in patch_size]
        transforms += [
            SpatialPadd(keys=keys, allow_missing_keys=True, spatial_size=pre_patch_size),  # Added in case img smaller than patch
            RandSpatialCropd(keys=keys, allow_missing_keys=True, roi_size=pre_patch_size, random_size=False),
        ]

    if spatial_augmentations:
        transforms += [
            RandFlipd( keys=keys, allow_missing_keys=True,  spatial_axis=[0], prob=0.10, ),
            RandFlipd( keys=keys, allow_missing_keys=True,  spatial_axis=[1], prob=0.10, ),
            RandFlipd( keys=keys, allow_missing_keys=True, spatial_axis=[2], prob=0.10, ),
            RandRotate90d( keys=keys, allow_missing_keys=True, prob=0.10, max_k=3, ),
            RandZoomd( keys=keys, allow_missing_keys=True, prob=0.10, padding_mode='constant'),
            RandRotated( keys=keys, allow_missing_keys=True, prob=0.10, range_x=0.1, range_y=0.1, range_z=0.1, padding_mode='zeros'),
        ]

    if spatial_augmentations and (patch_size is not None):
        transforms += [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=patch_size),]

    if (not spatial_augmentations) and (patch_size is not None):
        transforms += [
            SpatialPadd(keys=keys, allow_missing_keys=True, spatial_size=patch_size),  # Added in case img smaller than patch
            RandSpatialCropd(keys=keys, allow_missing_keys=True, roi_size=patch_size, random_size=False),
            ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=patch_size),
        ]

    if preserve_zero_background and intesity_augmentations:
        transforms += [GetForegroundMaskd(keys=["image"]),]

    if intesity_augmentations:
        transforms += [
            RandGaussianNoised( keys=["image"], prob=0.1, std=0.01 ),
            RandBiasFieldd( keys=["image"], coeff_range=(0.0, 0.3), prob=0.1, ),
            RandGaussianSmoothd( keys=["image"], prob=0.1 ),
            RandAdjustContrastd( keys=["image"], prob=0.1,  gamma=(0.5, 1.5) ),
            ScaleIntensityRanged( keys=["image"], a_min= 0.0, a_max=1, b_min=0, b_max=1, clip=True) # Clip to avoid overflow
        ]

    if normalize and (normalize_subtrahend_divisor is None):
        transforms += [ApplyIntensityMeandStdd(keys=['image'],nonzero=normalize_nonzero)]

    elif normalize and (normalize_subtrahend_divisor is not None):
        transforms += [NormalizeIntensityd(keys=['image'],
                                           subtrahend=normalize_subtrahend_divisor[0],
                                           divisor=normalize_subtrahend_divisor[1],
                                           nonzero=normalize_nonzero)]


    if preserve_zero_background and intesity_augmentations:
        transforms += [ApplyForegroundMaskd(keys=["image"]),]

    transforms += [CastToTyped( keys = ["image"],dtype = torch.float )]

    return transforms


def get_dataset(datasets,
                dataset_section="training",
                image_only=False,
                dataset_kwargs={"cache_rate":0, "num_workers":8},
                transforms_kwargs={},
                patch_size=(256, 256, 256),
                fpi_kwargs=None,
                ):

    # If it's a list of dictionaries assumes that it's providing already datalist
    if isinstance(datasets,list) and isinstance(datasets[0],dict):
        datalist = datasets
    # Otherwise loads the datalist from the dataset
    elif isinstance(datasets,(tuple,list)):
        datalist = []
        for i in datasets:
            datalist.extend(load_decathlon_datalist(i, False, dataset_section))
    elif isinstance(datasets,(str,PathLike)):
        datalist = load_decathlon_datalist( datasets, False, dataset_section )
    else:
        raise ValueError("Unknown dataset type")

    keys = ["image","label"] if not image_only else ["image"]
    transforms = get_transforms(keys=keys, patch_size = patch_size, in_memory= dataset_kwargs["cache_rate"] > 0,
                                **transforms_kwargs)

    if fpi_kwargs is not None:
        transforms += [FPIAnomalyd( image_key = "image", fp_key = "fp", mask_key = "mask",image_size = patch_size,
                                    **fpi_kwargs)]
    transforms = Compose(transforms)

    ds = CacheDataset(
        data = datalist,
        transform = transforms,
        **dataset_kwargs
    )

    return ds


