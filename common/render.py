import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import renderpy
from tqdm import tqdm

# PyPI hosts an unrelated package called "renderpy" (a render CLI). Installing that one imports
# fine and then dies here with `module 'renderpy' has no attribute 'Render'`, so check the API.
if not hasattr(renderpy, "Render"):
    print(
        f"The installed renderpy ({getattr(renderpy, '__file__', '?')}) is NOT liu115/renderpy: "
        "it has no `Render` class. It is the unrelated PyPI package of the same name.\n"
        "  pip uninstall renderpy\n"
        '  pip install "renderpy @ git+https://github.com/liu115/renderpy"'
    )
    sys.exit(1)

from common.utils.colmap import read_model
from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_yaml_munch, read_txt_list


def get_scene_ids(cfg):
    """
    Scene selection, in the same precedence as the rest of the toolkit:
    scene_list_file > scene_ids > splits.
    """
    if cfg.get("scene_list_file"):
        return read_txt_list(cfg.scene_list_file)
    if cfg.get("scene_ids"):
        return list(cfg.scene_ids)
    if cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            scene_ids += read_txt_list(Path(cfg.data_root) / "splits" / f"{split}.txt")
        return scene_ids
    raise ValueError(
        "No scenes selected: set one of scene_list_file, scene_ids or splits in the config."
    )


def get_frame_filter(scene, device, cfg):
    """
    Optional per-scene frame whitelist, so a scene can be rendered for a subset of its frames
    instead of every COLMAP image.

    Two ways to provide it:
      * cfg.frame_list_file: a path to a text file (one frame name or stem per line), or
      * cfg.frame_list_name: a file name looked up inside the device folder of each scene,
        e.g. "frames.txt" -> <scene>/iphone/frames.txt

    Returns a set of frame stems (e.g. {"frame_000000", ...}) or None for "render everything".
    """
    list_path = None
    if cfg.get("frame_list_file"):
        list_path = Path(cfg.frame_list_file)
    elif cfg.get("frame_list_name"):
        device_dir = scene.iphone_data_dir if device == "iphone" else scene.dslr_dir
        candidate = Path(device_dir) / cfg.frame_list_name
        list_path = candidate if candidate.is_file() else None

    if list_path is None:
        return None
    if not list_path.is_file():
        print(f"WARNING: frame list not found, rendering all frames: {list_path}")
        return None

    stems = {Path(line.strip()).stem for line in read_txt_list(list_path) if line.strip()}
    print(f"Frame list {list_path} -> {len(stems)} frames")
    return stems


def main(args):
    cfg = load_yaml_munch(args.config_file)

    scene_ids = get_scene_ids(cfg)

    # NOTE: this used to compute output_dir and then ignore it, reading cfg.output_dir directly
    # further down, which crashed whenever output_dir was not set in the config.
    output_dir = cfg.get("output_dir")
    if output_dir is None:
        # default to data folder in data_root
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")
    if not render_devices:
        raise ValueError("Nothing to render: set render_iphone and/or render_dslr to true.")

    near = cfg.get("near", 0.05)
    far = cfg.get("far", 20.0)
    write_rgb = cfg.get("render_rgb_images", True)  # the RGB renders are large and often unused
    skip_existing = cfg.get("skip_existing", True)  # makes the script resumable
    verbose = cfg.get("verbose", False)

    num_rendered, num_skipped, failed_scenes = 0, 0, []

    # go through each scene
    for scene_id in tqdm(scene_ids, desc="scene"):
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")

        mesh_path = Path(scene.scan_mesh_path)
        if not mesh_path.is_file():
            print(f"WARNING: no mesh for {scene_id}, skipping: {mesh_path}")
            failed_scenes.append((scene_id, "missing mesh"))
            continue

        try:
            render_engine = renderpy.Render()
            render_engine.setupMesh(str(mesh_path))
        except Exception as e:
            # renderpy needs a working offscreen GL context; on headless/NVIDIA boxes this is
            # usually a missing EGL vendor library rather than a problem with the mesh.
            print(
                f"ERROR: could not initialize renderpy for {scene_id}: {e!r}\n"
                "  If this is an EGL/OpenGL error: install libEGL (apt install libegl1 libgl1) and "
                "make sure your driver's EGL vendor JSON exists (NVIDIA: "
                "/usr/share/glvnd/egl_vendor.d/10_nvidia.json). To force it:\n"
                "    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
            )
            failed_scenes.append((scene_id, f"renderpy init: {e!r}"))
            continue

        for device in render_devices:
            colmap_dir = Path(scene.dslr_colmap_dir if device == "dslr" else scene.iphone_colmap_dir)
            if not colmap_dir.is_dir():
                print(f"WARNING: no {device} colmap model for {scene_id}, skipping: {colmap_dir}")
                failed_scenes.append((scene_id, f"missing {device} colmap"))
                continue

            cameras, images, points3D = read_model(colmap_dir, ".txt")
            if len(cameras) > 1:
                # ~25% of ScanNet++ iPhone models have more than one camera (the intrinsics change
                # mid-capture). The stock script asserts here; instead, set up the right camera per
                # image and only re-upload it when it actually changes.
                print(f"{scene_id}/{device}: {len(cameras)} cameras in the colmap model, "
                      f"switching intrinsics per image")

            def setup_camera(camera):
                fx, fy, cx, cy = camera.params[:4]
                params = camera.params[4:]
                render_engine.setupCamera(
                    camera.height, camera.width,
                    fx, fy, cx, cy,
                    camera.model,
                    params,  # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
                )

            active_camera_id = None
            keep_stems = get_frame_filter(scene, device, cfg)

            rgb_dir = output_dir / scene_id / device / "render_rgb"
            depth_dir = output_dir / scene_id / device / "render_depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            if write_rgb:
                rgb_dir.mkdir(parents=True, exist_ok=True)

            for image_id, image in tqdm(images.items(), f"Rendering {device} images", leave=False):
                # image.name can carry a relative path in some models; keep only the file name
                image_name = Path(image.name).name
                image_stem = Path(image_name).stem

                if keep_stems is not None and image_stem not in keep_stems:
                    continue

                rgb_path = rgb_dir / image_name
                depth_path = depth_dir / f"{image_stem}.png"
                if skip_existing and depth_path.is_file() and (not write_rgb or rgb_path.is_file()):
                    num_skipped += 1
                    continue

                if image.camera_id != active_camera_id:
                    setup_camera(cameras[image.camera_id])
                    active_camera_id = image.camera_id

                world_to_camera = image.world_to_camera
                rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
                rgb = rgb.astype(np.uint8)
                # Make depth in mm and clip to fit 16-bit image
                depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)

                if write_rgb:
                    imageio.imwrite(str(rgb_path), rgb)
                imageio.imwrite(str(depth_path), depth)
                num_rendered += 1

                if verbose:
                    print(f"{scene_id}/{device}/{image_name}: "
                          f"depth {depth.min() / 1000:.2f}-{depth.max() / 1000:.2f} m")

            if keep_stems is not None:
                missing = keep_stems - {Path(i.name).stem for i in images.values()}
                if missing:
                    print(f"WARNING: {len(missing)} requested frames are not in the {device} colmap "
                          f"model of {scene_id} and were not rendered, e.g. {sorted(missing)[:5]}")

    print(f"Rendered {num_rendered} images, skipped {num_skipped} existing ones.")
    if failed_scenes:
        print(f"{len(failed_scenes)} scene/device combinations failed:")
        for scene_id, reason in failed_scenes:
            print(f"  {scene_id}: {reason}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)