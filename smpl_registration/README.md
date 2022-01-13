## Registering SMPL to scans, kinect point clouds

### Fit SMPLH to scans
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

### Fit SMPLH+D model to scans


### Fit SMPLH model to Kinect point clouds
TODO