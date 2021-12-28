import json
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
import pyopenpose as op


def main(args):
    # custom params for the model (refer to include/openpose/flags.hpp for more parameters)
    op_params = dict()
    op_params["model_folder"] = "/openpose/models/"
    op_params["net_resolution"] = "720x480"
    op_params["scale_number"] = 4
    op_params["scale_gap"] = 0.25

    # start OpenPose
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(op_params)
    op_wrapper.start()

    # create list of paths to images
    image_folders = sorted(list(args.data_path.glob(f"*/t*/rgb")))

    for image_folder in tqdm(image_folders, ncols=80):
        # input file structure is <seq. name>/<time stamp>/rgb/*.png
        # results will be stored in <seq. name>/<time stamp>/2D_pose.json
        folder_tree = image_folder.relative_to(args.data_path)
        folder_tree = folder_tree.parent  # remove "rgb' folder
        result_path = args.results_path / folder_tree / "2D_pose.json"

        # skip if results already exist
        if not args.force and result_path.is_file():
            continue

        # create necessary directories
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # list input images
        image_paths = []
        for ext in ["jpg", "jpeg", "png"]:
            image_paths.extend(list(image_folder.glob(f"*.{ext}")))
        image_paths = sorted(image_paths)

        # create input Datums and get predictions
        op_vector_datum = []
        for image_path in image_paths:
            op_datum = op.Datum()
            image = cv2.imread(str(image_path))
            op_datum.cvInputData = image

            op_wrapper.emplaceAndPop(op.VectorDatum([op_datum]))
            op_vector_datum.append(op_datum)

        # convert OP datum to internal results format
        # results structure: {<filename>: [<list with OP predictions>]}
        results = {}
        for image_path, datum in zip(image_paths, op_vector_datum):
            filename = image_path.stem
            keypoints_2d = datum.poseKeypoints
            keypoints_2d = keypoints_2d.tolist() if keypoints_2d is not None else []

            results[filename] = keypoints_2d

            # optionally save visualisations
            if args.visualize:
                vis_result_path = result_path.parent / "2D_pose_vis" / image_path.name
                vis_result_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_result_path), datum.cvOutputData)

        # save results
        with result_path.open('w') as fp:
            json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenPose predictor.")
    parser.add_argument("--data-path", "-d", type=str,
                        help="Path to folder with data")
    parser.add_argument("--results-path", "-r", type=str,
                        help="Path to folder for results")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force overwriting existing results (default: False)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Save visualizations (default: False)")
    args = parser.parse_args()

    # Convert str to Path for convenience
    args.data_path = Path(args.data_path)
    args.results_path = Path(args.results_path)

    main(args)
