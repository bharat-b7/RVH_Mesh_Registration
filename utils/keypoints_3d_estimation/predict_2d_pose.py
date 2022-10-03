"""
use wrapper without the patch
"""

import json
import argparse
from pathlib import Path
import sys, os
# PYOPENPOSE_DIR = "/BS/xxie2020/work/software/openpose/build/python"
PYOPENPOSE_DIR = "/BS/xxie2020/work/software/openpose-cv4.5/build/python"
OPENPOSE_MODEL_DIR = "/BS/xxie2020/work/software/openpose/models"
sys.path.append(PYOPENPOSE_DIR)
import cv2
import numpy as np
# import pyopenpose as op
from openpose import pyopenpose as op

def preset_params(args):
    params = dict()

    params["body"] = 1 if 'b' in args.mode else 0
    params["model_folder"] = OPENPOSE_MODEL_DIR

    if 'h' in args.mode:
        params["hand"] = True
        # params["hand_net_resolution"] = "1312x736"
        params["hand_scale_number"] = 6
        params["hand_scale_range"] = 0.4
        params["hand_render_threshold"] = 0.01
    else:
        params["hand"] = False

    if 'f' in args.mode:
        params["face"] = True
        params["face_net_resolution"] = "480x480"
        params["face_render_threshold"] = 0.01
    else:
        params["face"] = False

    return params


def filter_background_detections(detections):
    if detections is not None and detections.ndim == 3:
        mean_confidence = np.mean(detections[:, :, 2], axis=1)
        index = np.argmax(mean_confidence)

        return detections[index].tolist()
    else:
        return []


def main(args):
    # custom params for the model (refer to include/openpose/flags.hpp for more parameters)
    op_params = dict()
    # op_params["model_folder"] = "/openpose/models/"
    op_params["model_folder"] = "/BS/xxie2020/work/software/openpose/models"
    op_params["net_resolution"] = "720x480"
    op_params["scale_number"] = 3
    op_params["scale_gap"] = 0.25
    op_params.update(preset_params(args))

    # start OpenPose
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(op_params)
    op_wrapper.start()

    # list input images
    input_image_paths = []
    for ext in ["jpg", "jpeg", "png"]:
        input_image_paths.extend(list(args.input_folder.glob(f"*.{ext}")))
    input_image_paths = sorted(input_image_paths)

    # create input Datums and get predictions
    op_vector_datum = []
    for input_image_path in input_image_paths:
        op_datum = op.Datum()
        image = cv2.imread(str(input_image_path))
        op_datum.cvInputData = image

        op_wrapper.emplaceAndPop(op.VectorDatum([op_datum]))
        op_vector_datum.append(op_datum)

    # convert OP datum to internal results format
    # results structure: {<filename>: [<list with OP predictions>]}
    results = {}
    for input_image_path, datum in zip(input_image_paths, op_vector_datum):
        print('Run openpose on', input_image_path)
        input_name = input_image_path.stem
        # results[input_name] = {
        #     "pose_keypoints_2d": filter_background_detections(datum.getPoseKeypoints()),
        #     "face_keypoints_2d": filter_background_detections(datum.getFaceKeypoints()),
        #     "hand_left_keypoints_2d": filter_background_detections(datum.getHandKeypointsL()),
        #     "hand_right_keypoints_2d": filter_background_detections(datum.getHandKeypointsR())
        # }

        # original op wrapper
        results[input_name] = {
            "pose_keypoints_2d": filter_background_detections(datum.poseKeypoints),
            "face_keypoints_2d": filter_background_detections(datum.faceKeypoints),
            "hand_left_keypoints_2d": filter_background_detections(datum.handKeypoints[0]),
            "hand_right_keypoints_2d": filter_background_detections(datum.handKeypoints[1])
        }

        # optionally save visualisations
        if args.visualize:
            vis_result_path = args.results_folder / "2D_pose_vis" / f"{input_name}.jpg"
            vis_result_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_result_path), datum.cvOutputData)

    # save results
    args.results_folder.mkdir(parents=True, exist_ok=True)
    with (args.results_folder / "2D_pose.json").open('w') as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenPose predictor.")
    parser.add_argument("input_folder", type=Path,
                        help="Path to folder with data")
    parser.add_argument("--results-folder", "-r", type=Path, default=None,
                        help="Path to folder for results, by default the results "
                             "are saved to the same folder (default: None)")
    parser.add_argument("--mode", "-m", type=str, choices=['b', 'h', 'f'], nargs='+', default=['b', 'h', 'f'],
                        help="Switching between detecting body, hand, and face joints, "
                             "modes can be combined (default: b h)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Save visualizations (default: False)")
    args = parser.parse_args()

    if args.results_folder is None:
        args.results_folder = args.input_folder

    main(args)
