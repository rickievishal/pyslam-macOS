# Running pySLAM on an Apple Silicon Mac

This guide documents the KITTI 06 video and dense-mapping setup used in this
checkout. It also includes the separate optional live-webcam setup. The commands
below are for macOS on Apple Silicon; other platforms and Python versions may
need different native dependencies.

## What the KITTI video setup does

The default [`config.yaml`](./config.yaml) selects the bundled monocular KITTI
06 video (`data/videos/kitti06/video_color.mp4`) and its timestamps and
ground-truth trajectory. Its camera and feature settings come from
[`settings/KITTI04-12.yaml`](./settings/KITTI04-12.yaml).

The run combines:

- OpenCV ORB features for sparse visual tracking and mapping.
- DBOW3 loop detection for loop closing and relocalization.
- Depth Anything V2 VITS with the outdoor Virtual KITTI metric-depth checkpoint
  to estimate depth for each monocular video frame.
- The voxel-grid volumetric integrator to build and display a dense colored
  point cloud.

This is estimated depth, not a depth sensor or stereo measurement. Its scale and
geometry can be inaccurate. Dense integration also depends on SLAM continuing
to estimate camera poses: if tracking is lost and relocalization fails, the
camera stops contributing reliable keyframes and the dense map can appear to
stop growing. In the viewer, use the **Draw Dense Map** checkbox to show or hide
the dense result.

## Clone and prepare the environment

Install Xcode Command Line Tools and Homebrew if they are not already present:

```bash
xcode-select --install
```

Then install the native build tools and clone the repository with its submodules:

```bash
brew install cmake eigen tbb opencv
git clone --recurse-submodules https://github.com/luigifreda/pyslam.git
cd pyslam
```

Use Python 3.11.9 or newer within the Python 3.11 series. The project declares
Python `>=3.11.9`; the working setup for this guide used Python 3.11.9:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  numpy opencv-python torch torchvision open3d \
  kornia==0.8.2 kornia_moons==0.2.9 pybind11 pyyaml scipy \
  matplotlib pyqt5 pyqtgraph PyOpenGL glfw rerun-sdk pytest ujson \
  gdown evo numba trimesh munch plyfile fast-pytorch-kmeans PyGLM \
  quaternion mcap mcap-ros1-support mcap-ros2-support timm einops
```

PyTorch uses Apple's MPS backend when the installed build and Mac support it;
otherwise inference falls back to CPU. You can check MPS availability with:

```bash
.venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())'
```

## Build the native modules used by this profile

The dense integrator and DBOW3 loop detector each need a compiled native
extension. Build them in the new checkout; do not copy a `.so` from another
machine or Python version.

Build the volumetric extension:

```bash
cmake -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DEigen3_DIR="$(brew --prefix eigen)/share/eigen3/cmake" \
  -DTBB_DIR="$(brew --prefix tbb)/lib/cmake/TBB" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build cpp/build --target volumetric --parallel 4
```

Build the DBOW3 Python binding using the virtual environment's Python:

```bash
Python3_EXECUTABLE="$PWD/.venv/bin/python" ./thirdparty/pydbow3/build.sh
```

The first DBOW3 run may download the ORB vocabulary (about 145 MB). Let that
download finish. The optional Pangolin viewer binding is needed to use the
interactive 3D viewer. Build it as follows if `pypangolin` is not importable:

```bash
cmake -S thirdparty/pangolin -B thirdparty/pangolin/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_PANGOLIN_LIBREALSENSE=OFF \
  -DBUILD_PANGOLIN_LIBREALSENSE2=OFF \
  -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF \
  -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBOPENEXR=OFF \
  -DBUILD_PANGOLIN_LZ4=OFF
cmake --build thirdparty/pangolin/build --parallel 8
cmake --build cpp/build --target glutils --parallel 4
```

## Download the depth checkpoint

The KITTI profile needs this checkpoint, which is not included in a fresh clone.
Download the VITS outdoor metric-depth weights to the exact path expected by
the estimator:

```bash
mkdir -p thirdparty/depth_anything_v2/metric_depth/checkpoints
curl -L \
  'https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true' \
  -o thirdparty/depth_anything_v2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vits.pth
```

The video itself, timestamps, and ground truth are already in the repository.

## Run the KITTI video

Run from the repository root. This command selects the Python SLAM core, which
avoids mixing the Homebrew-linked C++ OpenCV helpers with pip's `cv2` on macOS:

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
PYSLAM_USE_CPP=false \
MPLCONFIGDIR="$PWD/.matplotlib" \
./.venv/bin/python main_slam.py -c config.yaml
```

`KMP_DUPLICATE_LIB_OK=TRUE` is a macOS workaround for duplicate OpenMP runtimes
in some dependency combinations, not a general fix for incompatible native
libraries. Do not use it as a substitute for rebuilding extensions for the
active Python and architecture.

The active video profile enables front-end depth estimation and volumetric
integration in `config.yaml`; `main_slam.py` selects Depth Anything V2 with a
20 m maximum depth for this profile. To change the input video, timestamps, or
camera calibration, update the `VIDEO_DATASET` section and select a matching
camera settings file. Do not reuse KITTI calibration for an unrelated video.

## Optional live-webcam path

The webcam uses separate configs and launcher; it does not change the KITTI
video profile above. The launcher defaults to camera index 0, 640x480 at 30 FPS,
and no semantic segmentation:

```bash
./run_macbook_webcam_slam.sh
```

Grant camera access to the terminal application in **System Settings >
Privacy & Security > Camera**. You can override the camera and capture size:

```bash
CAMERA_ID=0 WIDTH=640 HEIGHT=480 FPS=30 ./run_macbook_webcam_slam.sh
```

The webcam profile uses the indoor Hypersim VITS checkpoint and approximate
initial calibration in `settings/MACBOOK_M4_WEBCAM.yaml`. Calibrate the actual
camera before relying on its geometry:

```bash
./calibrate_macbook_webcam.py --camera-id 0 --cols 9 --rows 6 --square-size 25
```

The `cols` and `rows` are the checkerboard's inner-corner counts; capture at
least 10 views at varied positions. The webcam launcher requires the Pangolin
and `glutils` viewer modules described above.

Semantic mapping is opt-in and off by default. Enable it with the launcher
switch:

```bash
./run_macbook_webcam_slam.sh --semantics
```

The semantic profile uses torchvision DeepLabV3 MobileNetV3 with VOC class
labels. Per-pixel class predictions are attached to keypoints, fused into map
points, and included in the semantic voxel map. This is class-label semantic
segmentation: it does not produce the open-vocabulary object detections or
distinct instance masks shown in some upstream examples. To disable semantics
explicitly, use `--no-semantics`; the equivalent environment toggle remains
available as `SEMANTIC_ENABLED=true|false`. Semantic inference adds computation
and can reduce the live frame rate.

## Troubleshooting

- **Missing `depth_anything_v2_metric_vkitti_vits.pth`:** download the checkpoint
  above and verify that the filename and directory match exactly.
- **DBOW3 `Vocabulary` or `pydbow3` API error:** rebuild with the command above
  using the active `.venv` Python. A stale extension built for another Python
  version or architecture will not work.
- **Dense integration unavailable:** build the `volumetric` target and verify
  that the `.venv` can import `open3d`.
- **The map stops growing:** check the tracking state and relocalization
  messages. If the camera pose is lost, dense integration cannot reliably add
  new geometry. Move the camera smoothly, avoid fast turns and motion blur, and
  retain textured, well-lit scenery in view.
- **Optional-backend import warnings:** warnings for unused extractors,
  segmentation models, depth estimators, or Gaussian-splatting support are not
  necessarily fatal. Install those optional dependencies only if selecting
  those features; errors for ORB, DBOW3, the selected depth model, or
  volumetric integration mean a required part of this profile is unavailable.

## Notes and scope

This documents a working project configuration, not a guarantee that every
optional pySLAM backend builds on every Mac. The webcam calibration and MPS
availability depend on the individual Mac. The full upstream C++ SLAM core is
not required by the commands above; the video launcher intentionally selects
the Python core on macOS.
