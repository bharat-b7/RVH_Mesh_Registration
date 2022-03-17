"""
Script for rendering the input point cloud from multiple viewpoints using PyTorch3D.

Author: Ilya Petrov
"""
import sys
import argparse
import pickle as pkl
from pathlib import Path
from typing import Union

import cv2
import torch
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointsRasterizer,
    PointsRenderer,
    PointsRasterizationSettings,
    NormWeightedCompositor,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    MeshRenderer,
    MeshRasterizer,
)

sys.path.append(".")
from utils.keypoints_3d_estimation.io import load_data


def create_renderer(input_type: str, n_views: int = 10, image_size: int = 512, elevation: float = 5.0,
                    up=((0, 1, 0),), center=((0, 0, 0),), device: torch.device = "cpu"):
    # Get a batch of viewing angles
    azim = torch.linspace(-180, 180, n_views)

    # Initialize a camera
    # With world coordinates +Y up, +X left and +Z in
    R, T = look_at_view_transform(dist=2.0, elev=elevation, azim=azim, up=up, at=center)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    if input_type == "pointcloud":
        # Define the settings for rasterization and shading
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=0.005,
            points_per_pixel=10,
            bin_size=128
        )

        # Create a renderer by composing a rasterizer and a shader
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            compositor=NormWeightedCompositor()
        )
    elif input_type == "mesh":
        # Define the settings for rasterization and shading
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        # Place point lights
        lights = PointLights(ambient_color=[[0.7, 0.7, 0.7]],
                             diffuse_color=[[0.2, 0.2, 0.2]],
                             specular_color=[[0.1, 0.1, 0.1]],
                             location=[[0.0, 5.0, 0.0]],
                             device=device)

        # Create a Phong renderer by composing a rasterizer and a shader
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )
    else:
        raise RuntimeError(f"Unsupported input type {input_type}")

    # create dictionary with parameters to serialize
    renderer_parameters = {
        "image-size": image_size,
        "n-views": n_views,
        "R": R.cpu().numpy(),
        "T": T.cpu().numpy()
    }

    return renderer, renderer_parameters


def get_center(input_data: Union[Meshes, Pointclouds]):
    if isinstance(input_data, Meshes):
        vertices_list = input_data.verts_list()[0]
    elif isinstance(input_data, Pointclouds):
        vertices_list = input_data.points_list()[0]
    else:
        RuntimeError(f"Unsupported input type {type(input_data)}")

    center = vertices_list.mean(0)
    return torch.unsqueeze(center, 0)


def main(args):
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load data
    input_type, input_data = load_data(args.input_path, device, args.texture_path)

    # Get center location (to control camera orientation)
    center = get_center(input_data)

    # Create renderer
    renderer, renderer_parameters = create_renderer(input_type=input_type, n_views=args.n_views, image_size=args.image_size,
                                                    elevation=args.elevation, center=center, device=device, up=args.camera_up)

    # Perform batch rendering
    images = renderer(input_data.extend(args.n_views))

    # Save results in results_path/<input filename>/"render_<image #>.jpg"
    images = images[..., :3].cpu().numpy()
    args.results_path .mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        # Result is in [0..1] so we have to convert
        cv2.imwrite(str(args.results_path / f"{i:03d}.jpg"), (255 * image).astype(np.uint8))

    if not args.skip_cameras:
        with (args.results_path / "p3d_render_data.pkl").open("wb") as fp:
            pkl.dump(renderer_parameters, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for rendering the input mesh or scan "
                                     "from multiple viewpoints using PyTorch3D.")
    # Path to input / output
    parser.add_argument("input_path", type=Path,
                        help="Path to input data (mesh or point cloud)")
    parser.add_argument("--texture-path", "-t", type=Path, default=None,
                        help="Path to texture file, only applicable for meshes (default: None)")
    parser.add_argument("--results-path", "-r", type=Path,
                        help="Path to folder for results (\"<input filename>_renders\" folder with renderings "
                             "is created there)")
    # Rendering parameters
    parser.add_argument("--n-views", "-n", type=int, default=10,
                        help="Number of viewpoints to render the scan from (default: 10)")
    parser.add_argument("--image-size", "-s", type=int, default=512,
                        help="Result image size (SxS, default: 512)")
    parser.add_argument("--elevation", "-e", type=float, default=10,
                        help="Elevation of all created cameras (default: 10)")
    parser.add_argument("--camera-up", "-up", nargs=3, type=float, default=(0, 1, 0),
                        help="Direction of camera up (default: (0, 1, 0))")

    # Additional parameters
    parser.add_argument("--skip-cameras", "-sc", action="store_true",
                        help="Don't save camera parameters in the results folder (default: False)")

    args = parser.parse_args()
    args.results_path = args.results_path / f"{args.input_path.stem}_renders"
    args.camera_up = (args.camera_up, )

    main(args)
