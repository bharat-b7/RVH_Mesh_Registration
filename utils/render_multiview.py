"""
Script for rendering the input pointcloud from multiple viewpoints using PyTorch3D.

Author: Ilya Petrov
"""
import sys
import argparse
import pickle as pkl
from pathlib import Path

import cv2
import torch
import trimesh
import numpy as np
from pytorch3d import io
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointsRasterizer,
    PointsRenderer,
    PointsRasterizationSettings,
    AlphaCompositor,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV
)


def load_data(input_type: str, input_path: Path, device: torch.device = "cpu", texture_path: Path = None):
    if input_type == "pointcloud":
        pc = trimesh.load(input_path, process=False)
        verts = torch.Tensor(pc.vertices).to(device).unsqueeze(0)
        colors = torch.Tensor(pc.colors / 255).to(device).unsqueeze(0)

        input_data = Pointclouds(points=verts, features=colors)
    elif input_type == "mesh":
        # Load mesh based on file extension
        suffix = input_path.suffix
        if suffix == '.obj':
            verts, faces, aux = io.load_obj(input_path)

            texture = None
            if texture_path is not None:
                if texture_path.is_file():
                    # Load image
                    texture_image = cv2.imread(str(texture_path))
                    # It's important to convert image to float
                    texture_image = torch.from_numpy(texture_image.astype(np.float32) / 255)

                    # Extract representation needed to create Textures object
                    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
                    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
                    texture_image = texture_image[None, ...]  # (1, H, W, 3)

                    texture = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
                else:
                    Warning("No texture file found for provided .obj scan")

            # Initialise the mesh
            input_data = Meshes(verts=[verts], faces=[faces.verts_idx], textures=texture).to(device)
        elif suffix == '.ply':
            # Ilya: this part wasn't tested properly
            reader = io.IO()
            input_data = reader.load_mesh(input_path, device=device)
        else:
            raise RuntimeError(f"Unknown scan format {suffix}")
    else:
        raise RuntimeError(f"Unsupported input type {input_type}")

    return input_data


def create_renderer(input_type: str, n_views: int = 10, image_size: int = 512, elevation: float = 5.0,
                    up=((0, 1, 0),), device: torch.device = "cpu"):
    # Get a batch of viewing angles
    azim = torch.linspace(-180, 180, n_views)

    # Initialize a camera
    # With world coordinates +Y up, +X left and +Z in
    R, T = look_at_view_transform(dist=2.0, elev=elevation, azim=azim, up=up)
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
            compositor=AlphaCompositor()
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


def main(args):
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load data
    input_data = load_data(args.input_type, args.input_path, device, args.tedxture_path)

    # Create renderer
    renderer, renderer_parameters = create_renderer(input_type=args.input_type, n_views=args.n_views, image_size=args.image_size,
                                                    elevation=args.elevation, device=device)

    # Perform batch rendering
    images = renderer(input_data.extend(args.n_views))

    # Save results in results_path/<input filename>/"render_<image #>.jpg"
    images = images[..., :3].cpu().numpy()
    args.results_path .mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        # Result is in [0..1] so we have to convert
        cv2.imwrite(str(args.results_path / f"{i:03d}.jpg"), (255 * image).astype(np.uint8))

    if args.save_cameras:
        with (args.results_path / "pytorch3d_params_and_cameras.pkl").open("wb") as fp:
            pkl.dump(renderer_parameters, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for rendering the input mesh or scan "
                                     "from multiple viewpoints using PyTorch3D.")
    # Path to input / output
    parser.add_argument("input_path", type=Path,
                        help="Path to input data (mesh or pointcloud)")
    parser.add_argument("--input-type", "-t", choices=["mesh", "pointcloud"], type=str,
                        help="Type of input data.")
    parser.add_argument("--texture-path", type=Path, default=None,
                        help="Path to texture file, only applicable for meshes (default: None)")
    parser.add_argument("--results-path", type=Path,
                        help="Path to folder for results (\"<input filename>_renders\" folder with renderings "
                             "is created there)")
    # Rendering parameters
    parser.add_argument("--n-views", "-n", type=int, default=10,
                        help="Number of viewpoints to render the scan from (default: 10)")
    parser.add_argument("--image-size", "-s", type=int, default=512,
                        help="Result image size (SxS, default: 512)")
    parser.add_argument("--elevation", "-e", type=float, default=10,
                        help="Elevation of all created cameras (default: 10)")

    # Additional parameters
    parser.add_argument("--save-cameras", "-c", action="store_true",
                        help="Save camera parameters in the results folder (default: False)")

    args = parser.parse_args()
    args.results_path = args.results_path / f"{args.input_path.stem}_renders"

    main(args)
