
## Rendering and lifting keypoints to 3D
This documentation explains how to obtain 3D body keypoints for SMPL registration.

The scripts are located in ```utils/keypoints_3d_estimation```

**Contents**
1. [OpenPose setup](#openpose-setup)
1. [Multi-view rendering](#multi-view-rendering)
1. [2D pose prediction](#2d-pose-prediction)
1. [Lifting 2D pose to 3D](#lifting-2d-pose-to-3d)

### OpenPose setup
Here we provide instructions for building OpenPose library from source with our patch.

```
PYTHON_LIB_FOLDER="/usr/local/lib/python3.6/"
OPENPOSE_BUILD_DIR="/openpose/"

# Install CMake
wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt
rm cmake-3.16.0-Linux-x86_64.tar.gz

# Build OpenPose
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose && git submodule update --init --recursive --remote
# Our patch for OpenPose
git apply <path-to-the-repo>/assets/openpose.patch
# Generate and build
mkdir ${OPENPOSE_BUILD_DIR} && cd ${OPENPOSE_BUILD_DIR} && \
    /opt/cmake-3.16.0-Linux-x86_64/bin/cmake -DBUILD_PYTHON=ON .. && make -j 16

# Install Python bindings
cd ${OPENPOSE_BUILD_DIR}/python/openpose && make install
cp ${OPENPOSE_BUILD_DIR}python/openpose/pyopenpose.cpython-36m-x86_64-linux-gnu.so ${PYTHON_LIB_FOLDER}/dist-packages
cd ${PYTHON_LIB_FOLDER}/dist-packages && ln -s pyopenpose.cpython-36m-x86_64-linux-gnu.so pyopenpose
```

### Multi-view rendering
Script: ```utils/keypoints_3d_estimation/01_render_multiview.py```

Sample command:
```python utils/keypoints_3d_estimation/01_render_multiview.py ../data/mesh_1/scan.obj -t ../data/mesh_1/scan_tex.jpg -r ../data/mesh_1```

Script supports both meshes and point clouds

### 2D pose prediction
Script: ```utils/keypoints_3d_estimation/02_predict_2d_pose.py```

Sample command: ```python utils/keypoints_3d_estimation/02_predict_2d_pose.py ../data/mesh_1/scan_renders/ -r ../data/mesh_1 -v```

Optionally hand and face joints can be predicted. 

### Lifting 2D pose to 3D
Script: ```utils/keypoints_3d_estimation/03_lift_kepoints.py```

Sample command: ```python utils/keypoints_3d_estimation/03_lift_keypoints.py ../data/mesh_1/scan.obj -k2 ../data/mesh_1/2D_pose.json -r ../data/mesh_1/3D_pose.json -cam ../data/mesh_1/scan_renders/p3d_render_data.pkl -c ./config.yml```
