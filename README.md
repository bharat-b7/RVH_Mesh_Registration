# MPI Mesh registration repository
This repository collects methods to register SMPL model to point clouds or 3D scans.

#### Contents
1. [Dependencies](#a-namerun-enva-running-environment)
2. [Prepare model files](#a-nameprep-modela-prepare-model-files)
3. [Different registration methods](#a-namereg-methodsa-different-registration-methods)

## <a name="run-env"></a> Dependencies
Most dependencies are included in *requirement.txt* file, the following modules need to be installed manually:
1. MPI-IS Mesh library, see installation [here](https://github.com/MPI-IS/mesh).

## <a name="prep-model"></a> Prepare model files
We use SMPL+H model in this repository, for details about preparing all required model files, please check [here](docs/prep_smpl.md).

## <a name="reg-methods"></a> Different registration methods
We provide various methods for registering SMPL-H to scans or point clouds:
1. [Fit SMPLH to scans](#fit-smplh)
2. [Fit SMPLH+D to scans](#fit-smplh+d)
3. [Fit SMPLH to point clouds](#fit-smplh-pc)
4. [Fit SMPLH+D to point clouds using IP-Net](#fit-smplh-pc-ipnet)


### <a name="fit-smplh"></a> Fit SMPLH to scans
For more accurate registration, we recommend to first obtain 3D body keypoints from scans using [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and optimization. See details [here](docs/lift_kpts.md). 

With the model files and 3D keypoints ready, you can run fitting with:
```
python smpl_registration/fit_SMPLH.py [scan_path] [pose_file] [save_path] 
[-gender male/female] [-mr root path to SMPLH model]
```
### <a name="fit-smplh+d"></a> Fit SMPLH+D model to scans
Fitting SMPLH+D is based on fitting SMPLH, hence the command is very similar, except you can provide existing SMPLH parameters as input. 
```
python smpl_registration/fit_SMPLH+D.py [scan_path] [pose_file] [save_path] 
[-smpl_pkl existing SMPLH parameters] 
[-gender male/female] [-mr root path to SMPLH model]
```

### <a name="fit-smplh-pc"></a> Fit SMPLH model to Kinect point clouds
The fitting procedure is very similar to scan fitting. But Kinect point clouds are noisy and incomplete and the person pose captured by Kinects can be much more diverse than scans, we recommend to provide 3d pose estimation to initialize the SMPL model. These initial pose estimations can be obtained from monocular pose estimation methods, for example, [FrankMocap](https://github.com/facebookresearch/frankmocap).

Also you can obtain 3D joints following instructions [here](docs/lift_kpts.md).

Run fitting:
```
python smpl_registration/fit_SMPLH_pcloud.py [pc_path] [j3d_file] [pose_init] [save_path] 
[-gender male/female] [-mr root path to SMPLH model]
```

### <a name="fit-smplh-pc-ipnet"></a> Fit SMPLH+D model to scans using IP-Net 
This fitting is based on the [IP-Net project](https://github.com/bharat-b7/IPNet). You can download the pretrained IP-Net model [here](https://datasets.d2.mpi-inf.mpg.de/IPNet2020/IPNet_p5000_01_exp_id01.zip). The SMPLH model structure is the same as before.
Run fitting:
```
python smpl_registration/fit_SMPLH_IPNet.py [pc_path] [checkpoint path] [save path] 
[-gender male/female] [-mr root path to SMPLH model]
```

Cite us:
```
@inproceedings{bhatnagar2020ipnet,
    title = {Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction},
    author = {Bhatnagar, Bharat Lal and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision ({ECCV})},
    month = {aug},
    organization = {{Springer}},
    year = {2020},
}
```

## License

Copyright (c) 2020 Bharat Lal Bhatnagar, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction** paper in documents and papers that report on research using this Software.
