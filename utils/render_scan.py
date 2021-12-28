"""
Script for rendering the input scan from multiple viewpoints using PyTorch3D.

Author: Ilya Petrov
"""
import sys
import argparse
import pickle as pkl
from pathlib import Path

import cv2
import torch
import numpy as np
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader
)

sys.path.insert(0, '.')
from lib.io import load_scan


def create_renderer(n_views: int = 10, image_size: int = 512, device: torch.device = "cpu"):
    # Get a batch of viewing angles
    elev = 5
    azim = torch.linspace(-180, 180, n_views)

    # Initialize a camera
    # With world coordinates +Y up, +X left and +Z in
    R, T = look_at_view_transform(dist=2.0, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

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

    return renderer


def main(args):
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load data
    mesh = load_scan(args.scan_path, args.texture_path, device)

    # Create renderer
    renderer = create_renderer(args.n_views, args.image_size, device)

    # Perform bach rendering
    images = renderer(mesh.extend(args.n_views))

    # Save results in results_path/<scan filename>/"render_<image #>.jpg"
    images = images[..., :3].cpu().numpy()
    args.results_path .mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        # Result is in [0..1] so we have to convert
        cv2.imwrite(str(args.results_path / f"{i:03d}.jpg"), (255 * image).astype(np.uint8))

    if args.save_cameras:
        cameras = renderer.rasterizer.cameras
        with (args.results_path / "pytorch3d_params_and_cameras.pkl").open("wb") as fp:
            pkl.dump({"image-size": args.image_size,
                      "n-views": args.n_views,
                      "cameras": cameras,
                      "elev": 5,
                      "azim": torch.linspace(-180, 180, n_views),
                      "dist": 2.0}, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for rendering the input scan from multiple viewpoints using PyTorch3D.")
    # Path to input / output
    parser.add_argument("scan_path", type=str,
                        help="Path to input scan")
    parser.add_argument("--texture-path", type=str, default=None,
                        help="Path to texture file (default: None)")
    parser.add_argument("--results-path", type=str,
                        help="Path to folder for results (\"<scan_filename>_renders\" folder with renderings "
                             "is created there)")
    # Rendering parameters
    parser.add_argument("--n-views", "-n", type=int, default=10,
                        help="Number of viewpoints to render the scan from (default: 10)")
    parser.add_argument("--image-size", "-s", type=int, default=512,
                        help="Result image size (SxS, default: 512)")

    # Additional parameters
    parser.add_argument("--save-cameras", "-c", action="store_true",
                        help="Save camera parameters in the results folder (default: False)")

    args = parser.parse_args()
    # Convert str to Path for convenience
    args.scan_path = Path(args.scan_path)
    args.texture_path = Path(args.texture_path)
    args.results_path = Path(args.results_path) / f"{args.scan_path.stem}_renders"

    main(args)
