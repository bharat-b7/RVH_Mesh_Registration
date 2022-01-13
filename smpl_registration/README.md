## Registering SMPL to scans, kinect point clouds
This module contains scripts that can fit SMPLH or SMPLH+D models to 3D scans or point clouds captured by Kinects.

1. [Organizing SMPLH model files](#smplh-files)
2. [Fit SMPLH to scans](#fit-smplh)
3. [Fit SMPLH+D to scans](#fit-smplh+d)
4. [Fit SMPLH to point clouds](#fit-smplh-pc)

### <a name="smplh-files"></a> Organizing SMPLH model files 
smplh model file structure: 
```
|model root
|--grab             # folder containing hand priors computed from grab dataset
|----lh_prior.pkl
|----rh_prior.pkl
|--regressors       # folder for body, face, and hand regressors
|--SMPLH_female.pkl # SMPLH female model blending weights 
|--SMPLH_male.pkl
|--SMPLH_neutral.pkl
|--template         # folder for the template mesh files
```
### <a name="fit-smplh"></a> Fit SMPLH to scans
With the model files ready, you can run fitting with:
```
python smpl_registration/fit_SMPLH.py [scan_path] [pose_file] [save_path] 
[-gender male/female/neutral] [-mr root path to SMPLH model]
```
### <a name="fit-smplh+d"></a> Fit SMPLH+D model to scans
Fitting SMPLH+D is based on fitting SMPLH, hence the command is very similar, except you can provide existing SMPLH parameters as input. 
```
python smpl_registration/fit_SMPLH+D.py [scan_path] [pose_file] [save_path] 
[-smpl_pkl existing SMPLH parameters] 
[-gender male/female/neutral] [-mr root path to SMPLH model]
```

### <a name="fit-smplh-pc"></a> Fit SMPLH model to Kinect point clouds
The fitting procedure is very similar to scan fitting. But Kinect point clouds are noisy and incomplete and the person pose captured by Kinects can be much more diverse than scans, we recommend to provide 3d pose estimation to initialize the SMPL model. These initial pose estimations can be obtained from monocular pose estimation methods, for example, [FrankMocap](https://github.com/facebookresearch/frankmocap).

Run fitting:
```
python smpl_registration/fit_SMPLH_pcloud.py [pc_path] [j3d_file] [pose_init] [save_path] 
[-gender male/female/neutral] [-mr root path to SMPLH model]
```