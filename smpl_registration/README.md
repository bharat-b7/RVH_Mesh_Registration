## Registering SMPL to scans, kinect point clouds
This module contains scripts that can fit SMPLH or SMPLH+D models to 3D scans or point clouds captured by Kinects.

1. [Organizing SMPL model files](#smpl-files)
1. [Organizing SMPLH model files](#smplh-files)
1. [Fit SMPLH to scans](#fit-smplh)
1. [Fit SMPLH+D to scans](#fit-smplh+d)
1. [Fit SMPLH to point clouds](#fit-smplh-pc)
1. [Fit SMPLH+D to point clouds using IP-Net](#fit-smplh-pc-ipnet)

### <a name="smpl-files"></a> Organizing SMPL model files
```
|model root
|--SMPL_male.pkl
|--SMPL_female.pkl
|--SMPL_neutral.pkl
|--priors
|----body_prior.pkl 
|--regressors
|----body_25_openpose_joints.pkl
```

### <a name="smplh-files"></a> Organizing SMPLH model files  
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

### <a name="fit-smplh-pc-ipnet"></a> Fit SMPLH+D model to scans using IP-Net 
This fitting is based on the [IP-Net project](#https://github.com/bharat-b7/IPNet). You can download the pretrained IP-Net model [here](#https://datasets.d2.mpi-inf.mpg.de/IPNet2020/IPNet_p5000_01_exp_id01.zip). The SMPLH model structure is the same as before.
Run fitting:
```
python smpl_registration/fit_SMPLH_IPNet.py [pc_path] [checkpoint path] [save path] 
[-gender male/female/neutral] [-mr root path to SMPLH model]
```