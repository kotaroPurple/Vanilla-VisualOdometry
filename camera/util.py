
import numpy as np
from numpy.typing import NDArray


def camera_parameters_to_ndarray(
        fx: float, fy: float, cx: float, cy: float,
        d0: float, d1: float, d2: float, d3: float, d4: float) -> tuple[NDArray, NDArray]:
    camera_mat = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    distortion = np.array([d0, d1, d2, d3, d4])
    return camera_mat, distortion


def prepare_generating_xyz(
        width: int, height: int, fx: float, fy: float, cx: float, cy: float) \
        -> tuple[NDArray, NDArray]:
    # input two outputs to generate_xyz()
    # x
    x_data = np.arange(width, dtype=np.float64)
    x_data = np.tile(x_data, height).reshape(height, width)
    x_data = (x_data - cx) / fx
    # y
    y_data = np.arange(height, dtype=np.float64)
    y_data = np.repeat(y_data, width).reshape(height, width)
    y_data = (y_data - cy) / fy
    return x_data, y_data


def generate_xyz(x_over_z: NDArray, y_over_z: NDArray, z_data: NDArray) -> NDArray:
    # prepare_generating_xyz() generates x_over_z and y_over_z
    xyz = np.empty(x_over_z.shape + (3,), dtype=x_over_z.dtype)
    xyz[..., 0] = x_over_z * z_data
    xyz[..., 1] = y_over_z * z_data
    xyz[..., 2] = z_data
    return xyz
