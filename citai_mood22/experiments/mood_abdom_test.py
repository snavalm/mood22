from functools import partial
import torch
import os
import citai_mood22.network.network


### Uncomment if using wandb to log experiment  #####
# wandb_config = {'project':'[PROJECT NAME]', 'entity':'[WANDB USER]'}

patch_size = [160, 160, 160]

experiment_id = 'mood_abdom_soft_fold_0_test'

checkpoint_dir = './checkpoints/'
log_dir = './logs/'
mood_dataset_dir = '[PATH TO MOOD ABDOM TRAIN DATASET]'

train_dataset_config = {
    "datasets": [os.path.join(mood_dataset_dir+'dataset_fold_0.json')],
    "dataset_section": "training",
    "patch_size": patch_size,
    "transforms_kwargs": {
        "spacing": [1, 1, 1],
        "spatial_augmentations": True,
        "intesity_augmentations": True,
        "preserve_zero_background" : True,},
    "fpi_kwargs": {
        "fp_size": [128, 128, 128],
        "p_anomaly": 0.9,
        "alpha_range": [0.3, 1.0],
        "smooth_mask": True,
        "fp_augmentations": True,
        "mask_generators_prob_dict": {
            "RandomSquare": 1,
            "RandomMask": 1,
            "RandomSphere": 1
        },
        "anomaly_interpolation":'poisson', # 'linear' or 'poisson' for PII
        "mask_generators_kwarg_dict": {
            "RandomSquare": {"rotation_prob": 0.5, "min_anom_sizes": [10, 10, 10],
                             "no_mask_in_background": False},
            "RandomMask": {"dataset": "./masks_synthetic/dataset.json",
                           "no_mask_in_background": False},
            "RandomSphere": {"min_anom_size": 10, "no_mask_in_background": False}
        }, },
    "dataset_kwargs": {"cache_rate": 1, "num_workers": 12}  # Change cache setting if running out of memory, 0 is no cache
}

train_loader_config = {"batch_size":4,"num_workers":18, "persistent_workers": True,"shuffle": True}


cuda_devices = [0]
model = citai_mood22.network.network.UNet(in_channels=1, out_channels=1, )

max_iterations = 35000
eval_frequency = 5000

optimizer = partial(torch.optim.AdamW, lr=1e-4, weight_decay=1e-5)
scheduler = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=1e-3, total_steps=max_iterations)



#### Uncomment if generated a synthetic validation dataset: ####

# validation_dataset_config  = {
#     "datasets": "path_to_synthetic_dataset_json.json",
#     "dataset_section" : "validation",
#     "dataset_kwargs": {"cache_rate": 0, "num_workers": 12}
# }
# val_loader_config = {"batch_size":4,"num_workers":12}

#### Validation JSON example: ####

# {"numValidation": 251,
#  "validation": [
#      {
#          "image": "./ano_00007/ano_00007_image.nii.gz",
#          "label": "./ano_00007/ano_00007_label.nii.gz",
#          "type": "synthetic_additive_noise"
#      },
#      {
#          "image": "./ano_00014/ano_00014_image.nii.gz",
#          "label": "./ano_00014/ano_00014_label.nii.gz",
#          "type": "synthetic_additive_noise"
#      },
#      {
#          "image": "./ano_00036/ano_00036_image.nii.gz",
#          "label": "./ano_00036/ano_00036_label.nii.gz",
#          "type": "synthetic_additive_noise"
#      },
#  ]
#  }

