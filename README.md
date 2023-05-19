
## Improving synthetic anomaly based out-of-distribution with harder anomalies

## Citing 

If you use this implementation, please cite (paper coming soon):
```
@article{marimontMOOD2022,
  title={Improving synthetic anomaly based out-of-distribution with harder anomalies},
  author={Naval Marimont, Sergio and Tarroni, Giacomo},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snavalm/mood22},
  year={2023}
}
```

## Instructions

#### Copy dataset JSON specification files
We use monai pre-processing pipelines which require json files with the dataset information and training / validation 
splits. Examples of JSON files are provided in the `json_datalist` directory. Copy abdominal / brain JSON files to 
the respective MOOD dataset directories, i.e. the same directory where the MOOD `.nii.gz` files are.

#### [OPTIONAL] Generate a validation set
We used FPI [1] script provided [here](https://github.com/jemtan/FPI/blob/master/synthetic/example_synthesizing_outliers_mood.ipynb) 
to generate synthetic validation dataset. If you have a validation set create a JSON file for it such as:

```
{"numValidation": 251,
 "validation": [
     {
         "image": "./ano_00007/ano_00007_image.nii.gz",
         "label": "./ano_00007/ano_00007_label.nii.gz",
         "type": "synthetic_additive_noise"
     },
     {
         "image": "./ano_00014/ano_00014_image.nii.gz",
         "label": "./ano_00014/ano_00014_label.nii.gz",
         "type": "synthetic_additive_noise"
     },
     {
         "image": "./ano_00036/ano_00036_image.nii.gz",
         "label": "./ano_00036/ano_00036_label.nii.gz",
         "type": "synthetic_additive_noise"
     },
 ]
 }
```

If not using validation set, do not specify the `validation_dataset_config` key in the experiment configuration.

#### Define experiment configuration
Example of configuration is provided in [mood_abdom_test](citai_mood22/experiments/mood_abdom_test.py)

#### Train a Model

Run train_model.py providing the experiment configuration file as argument:
```
python train_model.py -e experiments.mood_abdom_test
```

#### Visualize anomaly generation process

Visualize the abdominal generation process using [visualize_anomalies.ipynb](visualize_anomalies.ipynb) notebook.


## Team

Sergio Naval Marimont, Giacomo Tarroni


## License

Improving synthetic anomaly based out-of-distribution with harder anomalies is relased under the MIT License.


[1]: Jeremy Tan, Benjamin Hou, James Batten, Huaqi Qiu, and Bernhard Kainz.: Foreign Patch Interpolation. Medical Out-of-Distribution Analysis Challenge at MICCAI. (2020)
