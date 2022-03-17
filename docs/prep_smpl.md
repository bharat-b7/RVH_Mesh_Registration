[comment]: <> (### Organizing model files)

[comment]: <> (```)

[comment]: <> (|model root)

[comment]: <> (|--SMPL_male.pkl)

[comment]: <> (|--SMPL_female.pkl)

[comment]: <> (|--SMPL_neutral.pkl)

[comment]: <> (|--priors)

[comment]: <> (|----body_prior.pkl )

[comment]: <> (|--regressors)

[comment]: <> (|----body_25_openpose_joints.pkl)

[comment]: <> (```)

### Organizing SMPL model files
To run the registration scripts, different files are required depending on whether you use SMPL or SMPL-H model:
```
|model root
|--SMPL_male.pkl # SMPL body male model, required if you use SMPL
|--SMPL_female.pkl
|--SMPL_neutral.pkl
|--SMPLH_female.pkl # SMPLH body models, required if you use SMPL-H  
|--SMPLH_male.pkl
|--priors             # folder containing body and hand pose priors
|----body_prior.pkl  # body prior, required both in SMPL and SMPL-H
|----lh_prior.pkl # hand priors, required if you use SMPL-H model 
|----rh_prior.pkl
|--regressors       # folder for body, face, and hand regressors, required both for SMPL and SMPL-H
|----body_25_openpose_joints.pkl
|----face_70_openpose_joints.pkl
|----hands_42_openpose_joints.pkl
```


#### SMPL or SMPL-H body models
For SMPL body model files `SMPL_*.pkl`, you can download from the [SMPL website](https://smpl.is.tue.mpg.de/download.php), once download and unzip finished, you can rename the model files in the following way to follow our convention:
```
basicmodel_f_lbs_10_207_0_v1.1.0.pkl -> SMPL_female.pkl
basicmodel_m_lbs_10_207_0_v1.1.0.pkl -> SMPL_male.pkl
basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl -> SMPL_neutral.pkl
```



For SMPL-H body model files `SMPLH_*.pkl`, you can download from the [SMPL-H website](https://mano.is.tue.mpg.de/index.html) and place them to your model root accordingly.

#### Priors
You can download our prebuilt priors in `assets/priors`. The body prior `body_prior.pkl` is required whether you use SMPL or SMPL-H. The hand priors `lh_prior.pkl` and `rh_prior.pkl` are required if you use SMPL-H model. 

The body prior was built from a subset of [AMASS](https://amass.is.tue.mpg.de/) dataset and hand priors were built from [GRAB](https://grab.is.tue.mpg.de/) dataset. 

Alternatively you can build your own priors from another dataset using the script `utils/build_prior.py`:
```angular2html
python utils/build_prior.py data_path out_path
```

#### Joint regressors
You can download these files from `assets/regressors`. 

Once these files are ready, you can change `SMPL_MODELS_PATH` in `config.yml` accordingly.