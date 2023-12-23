# Two Views Geometry
## Stereo Image Processing

This repository contains two Python scripts for stereo image processing. The scripts use OpenCV, NumPy, and Open3D libraries. The repository includes two files:

### File 1: `lab2_epipolar.py`

This script processes stereo images to compute the fundamental matrix.

#### Functions

- `normalize_transformation(points: np.ndarray) -> np.ndarray`: Computes a similarity transformation matrix that translates points, ensuring their center is at the origin and the average distance from the origin is sqrt(2).

- `homogenize(points: np.ndarray) -> np.ndarray`: Converts points to homogeneous coordinates.

- Image reading and keypoint detection using OpenCV's ORB detector.

- Matching keypoints and organizing them into matrices.

- Normalizing keypoints using the computed similarity transformations.

- Constructing a homogeneous linear equation to find the fundamental matrix.

- Obtaining the fundamental matrix and forcing it to have rank 2.

- Denormalizing the fundamental matrix.

#### Usage


1. Run the script python3 lab2_epipolar.py

### File 2: `lab2_stereo.py`

This script computes the depth map and generates a 3D point cloud from stereo images.

#### Functions

- Reading stereo images and computing the disparity map.

- Computing the depth of pixels with positive disparity.

- Computing normalized coordinates and 3D coordinates of pixels with positive disparity.

- Extracting color information for 3D points.

- Displaying the 3D point cloud using Open3D.

#### Usage

1. Run the script python3 lab2_stereo.py

### Prerequisites

- OpenCV
- NumPy
- Open3D

#### Installation

```bash
pip install opencv-python numpy open3d
