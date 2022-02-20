## Prepare SMPL+H model files
To run the registration scripts, the following file structures are required:
```
|model root
|--SMPLH_female.pkl # SMPLH female model
|--SMPLH_male.pkl
|--priors             # folder containing body and hand pose priors
|----body_prior.pkl 
|----lh_prior.pkl
|----rh_prior.pkl
|--regressors       # folder for body, face, and hand regressors
|----body_25_openpose_joints.pkl
|----face_70_openpose_joints.pkl
|----hands_42_openpose_joints.pkl
```
#### SMPL-H body models
Please download the body model files from the [official website](https://mano.is.tue.mpg.de/index.html) and place them to your model root accordingly.
#### Priors
You can download our prebuilt priors in `assets/priors`. The body prior was built from a subset of [AMASS](https://amass.is.tue.mpg.de/) dataset and hand priors were built from [GRAB](https://grab.is.tue.mpg.de/) dataset. 

Alternatively you can build your own priors from another dataset using the script `utils/build_prior.py`:
```angular2html
python utils/build_prior.py data_path out_path
```

#### Joint regressors
You can download these files from `assets/regressors`. 

Once these files are ready, you can change `SMPLH_MODELS_PATH` in `config.yml` accordingly.