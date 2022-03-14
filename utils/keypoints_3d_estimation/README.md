### Steps to run
```
python utils/keypoints_3d_estimation/render_multiview.py ../data/scan.obj -t ../data/scan_tex.jpg -r ../data/ -c
python ./utils/keypoints_3d_estimation/predict_2d_pose.py ../data/scan_renders/ -r ../data/ -v
python utils/keypoints_3d_estimation/lift_keypoints.py ../data/scan.obj -k2 ../data/2D_pose.json -r ../data/3D_pose.json -cam ../data/scan_renders/pytorch3d_params_and_cameras.pkl -c ./config.yml
```