import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from monai.data import (
    NumpyReader, NibabelReader
)

from monai.transforms import *
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor, PathLike
from typing import Any, Tuple, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Union
from monai.utils import fall_back_tuple
from monai.data.utils import get_random_patch, get_valid_patch_size

from monai.utils import ensure_tuple

import torch.nn as nn

## Custom Reader for RADChestCT
class RADChestCTReader(NumpyReader):
    def read(self,data):
        data = ensure_tuple(data)
        out = (np.load(d)['ct'].transpose(2,1,0) for d in data)
        return out

    def get_data(self, img):
        img, meta  = super().get_data(img)
        meta['space'] = 'LPS'
        meta['affine'] = np.array([[-0.8, 0., 0., 0.],
                                   [0., -0.8, 0., 0.],
                                   [0., 0., -0.8, 0.],
                                   [0., 0., 0., 1.]])


        return img, meta


class RandSpatialCropPreSpacing(Randomizable, Crop):
    """
    Like RandSpatialCrop but takes into consideration that the image will have the spacing adjusted.
    Avoids doing the spacing to the full image to later crop
    """

    def __init__(
            self,
            roi_size: Union[Sequence[int], int],
            spacing: Optional[Union[Sequence[int], int]] = None
    ) -> None:
        self.roi_size = roi_size
        self._size: Optional[Sequence[int]] = None
        self._slices: Tuple[slice, ...]
        self.spacing = torch.tensor(spacing) if spacing is not None else None

    def randomize(self, img_size: Sequence[int], affine) -> None:

        if self.spacing is not None:
            roi_size = torch.tensor(self.roi_size) * (self.spacing / torch.diag(affine)[:-1])
            roi_size = torch.ceil(roi_size).to(int)
        else:
            roi_size = self.roi_size

        self._size = fall_back_tuple(roi_size, img_size)

        valid_size = get_valid_patch_size(img_size, self._size)
        self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        if randomize:
            self.randomize(img.shape[1:], img.meta['affine'])
        if self._size is None:
            raise RuntimeError("self._size not specified.")
        return super().__call__(img=img, slices=self._slices)


class RandSpatialCropPreSpacingd(RandCropd):
    """
    Like RandSpatialCropd but takes into consideration that the image will have the spacing adjusted.
    Avoids doing the spacing to the full image to later crop
    """
    def __init__(
            self,
            keys: KeysCollection,
            roi_size: Union[Sequence[int], int],
            allow_missing_keys: bool = False,
            spacing: Optional[Union[Sequence[int], int]] = None
    ) -> None:
        cropper = RandSpatialCropPreSpacing(roi_size, spacing)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys)

    def randomize(self, img_size: Sequence[int], affine) -> None:
        if isinstance(self.cropper, Randomizable):
            self.cropper.randomize(img_size, affine)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        # the first key must exist to execute random operations
        self.randomize(d[self.first_key(d)].shape[1:], d[self.first_key(d)].meta['affine'])
        for key in self.key_iterator(d):
            kwargs = {"randomize": False} if isinstance(self.cropper, Randomizable) else {}
            d[key] = self.cropper(d[key], **kwargs)  # type: ignore
        return d

class GetForegroundMaskd(MapTransform):
    def __init__(self, keys = ['Image'], mask_key: str = 'ForegroundMask',
    ):
        super().__init__( keys )
        self.mask_key = mask_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        d[self.mask_key] = d[self.keys[0]] > 0
        return d

class ApplyForegroundMaskd(MapTransform):
    def __init__(self, keys = ['Image'], mask_key = 'ForegroundMask'):
        super().__init__( keys )
        self.mask_key = mask_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        for key in self.key_iterator( d ):
            d[key] =  d[key] * d[self.mask_key].float()
        return d


class ComputeIntensityMeandStdd(MapTransform):
    def __init__(self, keys = ['Image'], nonzero = False):
        super().__init__( keys )
        self.nonzero = nonzero

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        for key in self.key_iterator( d ):
            if self.nonzero:
                slices = d[key] != 0
            else:
                slices = torch.ones_like(d[key], dtype=torch.bool)

            d[key+'_mean'] =  d[key][slices].mean()
            d[key+'_std'] =  d[key][slices].std(unbiased=False)

        return d


class ApplyIntensityMeandStdd(MapTransform):
    def __init__(self, keys=['Image'], nonzero=False):
        super().__init__(keys)
        self.nonzero = nonzero

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.nonzero:
                slices = d[key] != 0
            else:
                slices = torch.ones_like(d[key], dtype=torch.bool)

            d[key][slices] = (d[key][slices] - d[key + '_mean']) / d[key + '_std']

        return d


class SetLabelNumd( MapTransform ):
    backend = SpatialCrop.backend

    def __init__(
            self,
            keys: KeysCollection = 'label',
            type_key: KeysCollection = 'type',
            map_type_to_label: Dict = {},
            allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__( keys, allow_missing_keys )
        self.map_type_to_label = map_type_to_label
        self.type_key = type_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict( data )

        label_no = 0 if d[self.type_key] not in self.map_type_to_label else self.map_type_to_label[d[self.type_key]]

        d['type_label'] = label_no

        for key in self.key_iterator( d ):
            d[key] =  d[key] * label_no
        return d




class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels=1, kernel_size=3, sigma=1., dim=3):
        super(GaussianSmoothing, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim
        self.kernel_size = kernel_size
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid( [torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing='ij')
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding = [i//2 for i in self.kernel_size])


# FPI stuff
import random
from monai.data import load_decathlon_datalist, CacheDataset


class MaskGenerator():
    def __init__(self,
                 patch_size,
                 no_mask_in_background = False,
                 rotation_prob = 0.0):
        self.patch_size = patch_size
        self.no_mask_in_background = no_mask_in_background
        self.rotation_prob = rotation_prob

    def generate_mask(self,*args,**kwargs):
        out = torch.ones(1,*self.patch_size)
        return out

    def random_rotation(self, mask):

        if random.random() < self.rotation_prob:
            rotator = Rotate(
                angle = (random.uniform(-3,3), random.uniform(-3,3), random.uniform(-3,3)),
                keep_size = True,
                mode = "nearest",
                padding_mode = "zeros",
                align_corners = False,
                dtype = mask.dtype,
            )
            mask = rotator( mask )

        return mask

    def __call__(self, *args, **kwargs):

        out = self.generate_mask(*args,**kwargs)
        out = self.random_rotation(out)

        if self.no_mask_in_background:
            assert 'image' in kwargs.keys(), 'image is required when calling mask generator'
            out[kwargs['image']==0] = 0
        return out

class RandomSquare(MaskGenerator):
    def __init__(self,
                 patch_size,
                 no_mask_in_background = False,
                 rotation_prob = 0.0,
                 min_anom_sizes=[10, 10, 10],
                 max_anom_sizes=None,
                 ):
        self.patch_size = patch_size
        self.no_mask_in_background = no_mask_in_background
        self.rotation_prob = rotation_prob
        self.min_anom_sizes = min_anom_sizes
        self.max_anom_sizes = max_anom_sizes if max_anom_sizes is not None else patch_size

    def generate_mask(self,*args,**kwargs):

        anom_size = [random.randint(a,b) for a,b in zip(self.min_anom_sizes,self.max_anom_sizes)]
        anom_center = [random.randint(a_s//2 , p_s - a_s // 2 ) for a_s, p_s in zip(anom_size,self.patch_size)]

        out = torch.zeros( 1, *self.patch_size )
        out[:, anom_center[0] - anom_size[0] // 2:anom_center[0] + anom_size[0] // 2,
        anom_center[1] - anom_size[1] // 2:anom_center[1] + anom_size[1] // 2,
        anom_center[2] - anom_size[2] // 2:anom_center[2] + anom_size[2] // 2] = 1

        return out

class RandomSphere(MaskGenerator):
    def __init__(self,
                 patch_size,
                 no_mask_in_background=False,
                 min_anom_size=10,
                 max_anom_size=None,
                 ):
        self.patch_size = patch_size
        self.no_mask_in_background = no_mask_in_background
        self.min_anom_size = min_anom_size
        self.max_anom_size = max_anom_size if max_anom_size is not None else min(patch_size)
        self.rotation_prob = 0.
        self.mesh = torch.stack( torch.meshgrid( [torch.linspace( 0, i, i ) for i in self.patch_size], indexing ='ij' ) )


    def generate_mask(self,*args,**kwargs):
        anom_size = random.randint( self.min_anom_size, self.max_anom_size )
        anom_center = [random.randint( anom_size // 2, p_s - anom_size // 2 ) for p_s in self.patch_size ]

        # Create a mask with the position of the anomaly
        mask_anom = (self.mesh - torch.tensor( anom_center )[:, None, None, None])**2
        mask_anom = mask_anom.sum( 0 ) <= (anom_size//2)**2
        mask_anom = mask_anom[None]

        return mask_anom.float()


class RandomMask(MaskGenerator):
    def __init__(self,
                 patch_size,
                 no_mask_in_background=False,
                 dataset=None,
                 num_workers=4,
                 zoom=None
                 ):
        self.patch_size = patch_size
        self.no_mask_in_background = no_mask_in_background
        self.rotation_prob = 0

        datalist = load_decathlon_datalist( dataset, True, 'training' )

        transforms = [
            LoadImaged( keys = ["label"],  ensure_channel_first = True, image_only=True ),
            Lambdad( keys = ['label'], func = lambda x: x > 0 ), # The mask can have more than one non-background classes
            CropForegroundd( keys = ["label"], source_key = "label" ),
            ]

        if zoom is not None:
            transforms.append(Zoomd( keys = ['label'], zoom = zoom, keep_size = False ),)

        transforms.extend([
            RandAffined( keys = ['label'], prob = 0.5, rotate_range = 2.,
                         shear_range = .5,
                         padding_mode = 'zeros',
                         scale_range = .3, mode = 'nearest' ),
            ResizeWithPadOrCropd( keys = ['label'],
                                  spatial_size = patch_size,
                                  ),
            ToTensord( keys = ["label"], ),]
        )

        transforms = Compose( transforms )

        self.ds = CacheDataset(
            data = datalist,
            transform = transforms,
            cache_rate = 1.,
            num_workers = num_workers
        )

    def generate_mask(self,*args,**kwargs):
        mask_anom = random.choice( self.ds )
        return mask_anom['label'].float()



map_mask_generator = {"RandomSquare":RandomSquare,
                      "RandomSphere":RandomSphere,
                      "RandomMask":RandomMask,
                      "MaskGenerator":MaskGenerator}



dict_smoothers = {k:GaussianSmoothing(1,k) for k in [3,5,7]}
dict_smoothers[1] = lambda x:x


class FPIAnomalyd( Randomizable, MapTransform ):

    def __init__(self,
                 image_key:KeysCollection = "image",
                 mask_key: KeysCollection = "mask",
                 fp_key: KeysCollection = "fp",
                 p_anomaly = 0.3,
                 alpha_range = [0,1],
                 label_binary = False,
                 fp_size = [64,64,64],
                 fp_augmentations=False,
                 mask_generators_prob_dict={},
                 mask_generators_kwarg_dict={},
                 size_cache = 50,
                 image_size = (256,256,64),
                 smooth_mask = False,
                 anomaly_interpolation = 'linear'
                 ):
        """
        Args:
            keys: keys of the corresponding items to be sampled from.
            anomaly_interpolation: interpolation method for the anomaly. Can be 'linear' or 'poisson'.
                                    If poisson please install pie-torch: https://pypi.org/project/pie-torch/
        """
        MapTransform.__init__( self, image_key, allow_missing_keys = True )
        self.p = p_anomaly
        self.image_key = image_key
        self.mask_key = mask_key
        self.fp_key = fp_key
        self.image_size = image_size

        self.alpha_range = alpha_range
        self.label_binary = label_binary

        self.size_cache = size_cache
        self.fp_size = fp_size
        self.fp_augmentations = fp_augmentations
        self.cache_patches = []

        self.anomaly_interpolation = anomaly_interpolation

        assert anomaly_interpolation in ['linear','poisson','none'], "Anomaly interpolation method not recognized. Please choose between 'linear' and 'poisson'."

        if anomaly_interpolation == 'poisson':
            try:
                self.pietorch = __import__('pietorch')
            except:
                raise ImportError("Please install pie-torch: https://pypi.org/project/pie-torch/")

        # Turn text into classes
        mask_generators_prob_dict = {map_mask_generator[k]:v for k,v in mask_generators_prob_dict.items()}
        mask_generators_kwarg_dict = {map_mask_generator[k]:v for k,v in mask_generators_kwarg_dict.items()}

        self.mask_generators_prob = mask_generators_prob_dict.values()
        self.mask_generators = [mg(patch_size=fp_size,**mask_generators_kwarg_dict[mg]) for mg in mask_generators_prob_dict.keys()]

        # Foreign patch transforms
        transforms  = [RandSpatialCrop(roi_size = fp_size,
                                       random_size = False)]

        if self.fp_augmentations:
            transforms.extend( [RandShiftIntensity( offsets = 0.2,
                                                    prob = 0.2 ),
                                RandGaussianNoise( std = 0.2,
                                                   prob = 0.2 ),
                                ScaleIntensityRange(
                                    a_min = 0.0,
                                    a_max = 1.0,
                                    b_min = 0.0,
                                    b_max = 1.0,
                                    clip = True, ),
                                RandRotate( 1, 1, 1,
                                            prob = 0.2 )] )

        self.patch_sampler = Compose(transforms)

        self.smooth_mask = smooth_mask
        self.smoother = dict_smoothers

    def push_patch_to_cache(self, patch):
        self.cache_patches.append( patch )
        # if cache is larger than textures_cache_len, get rid of some from the beginning
        while self.size_cache < len( self.cache_patches ):
            self.cache_patches.pop( 0 )

    def randomize(self,image):

        self.do_anomaly = random.random() < self.p

        if self.do_anomaly:
            # Pull a foreign patch from cache
            self.fp = self.cache_patches.pop( random.randrange( len( self.cache_patches ) ) )

            self.anom_center = torch.tensor([random.randint( a // 2, i - (a // 2))  for i, a in zip( self.image_size, self.fp_size )])
            self.alpha = random.random() * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]

            # Generate a mask
            mg = random.choices(self.mask_generators, weights = self.mask_generators_prob, k = 1)
            self.mask_anom = mg[0](
                image = image[:, self.anom_center[0] - self.fp_size[0] // 2:self.anom_center[0] + self.fp_size[0] // 2,
                        self.anom_center[1] - self.fp_size[1] // 2:self.anom_center[1] + self.fp_size[1] // 2,
                        self.anom_center[2] - self.fp_size[2] // 2:self.anom_center[2] + self.fp_size[2] // 2] )

            if self.smooth_mask:
                self.k_smooth = random.choice( [1, 1, 1, 1, 3, 3, 5, 7] )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict( data )

        inserted = 0
        while len(self.cache_patches) < self.size_cache:

            patch = self.patch_sampler(d[self.image_key])
            self.push_patch_to_cache(patch)

            inserted +=1

            # Avoid filling the whole cache with the same image initially
            if inserted >= 5:
                break

        # Generate set of patch centres and alpha for the interpolation
        self.randomize(d[self.image_key])

        # Initialize keys with Foreign Patch and Masks
        d[self.fp_key] = torch.zeros( 1, *self.image_size )
        d[self.mask_key] = torch.zeros( 1, *self.image_size )
        d['anom_center'] = torch.tensor([-1,-1,-1])

        if self.do_anomaly:

            # Create mask (location of anomaly) x alpha x pattern of anomaly
            d[self.mask_key][:, self.anom_center[0] - self.fp_size[0] // 2:self.anom_center[0] + self.fp_size[0] // 2,
            self.anom_center[1] - self.fp_size[1] // 2:self.anom_center[1] + self.fp_size[1] // 2,
            self.anom_center[2] - self.fp_size[2] // 2:self.anom_center[2] + self.fp_size[2] // 2] = self.alpha * self.mask_anom

            # Smooth the mask with avg pooling
            if self.smooth_mask:
                d[self.mask_key] = self.smoother[self.k_smooth](d[self.mask_key][None])[0]

            # Create fp
            d[self.fp_key][:, self.anom_center[0] - self.fp_size[0] // 2:self.anom_center[0] + self.fp_size[0] // 2,
            self.anom_center[1] - self.fp_size[1] // 2:self.anom_center[1] + self.fp_size[1] // 2,
            self.anom_center[2] - self.fp_size[2] // 2:self.anom_center[2] + self.fp_size[2] // 2] = self.fp

            d['anom_center'] = self.anom_center

            # Interpolation with foreign patch
            if self.anomaly_interpolation == 'linear':
                d[self.image_key] = d[self.image_key] * (1 - d[self.mask_key]) + d[self.fp_key] * d[self.mask_key]
            elif self.anomaly_interpolation == 'poisson':
                d[self.image_key].data = self.pietorch.blend(target=d[self.image_key].data,
                                                        source=d[self.fp_key].data,
                                                        mask=d[self.mask_key].data[0], # Blend expects no channel dimension
                                                        corner_coord=torch.tensor((0, 0, 0)),
                                                        mix_gradients=True, channels_dim=0)

            if self.label_binary:
                d[self.mask_key] = (d[self.mask_key] > 0).float()

        return d
