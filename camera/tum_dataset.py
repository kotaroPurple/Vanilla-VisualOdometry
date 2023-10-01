
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from numpy.typing import NDArray
# local
from camera.base import CameraBase
from camera.util import prepare_generating_xyz
from camera.util import generate_xyz
from camera.util import camera_parameters_to_ndarray
from images.util import read_image
from images.image_class import ImageWithFeature


# see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
# > The color and depth images are already pre-registered using the OpenNI driver from PrimeSense,
# > i.e., the pixels in the color and depth images correspond already 1:1.
@dataclass
class TumParameter:
    width: int = 640
    height: int = 480
    depth_min: float = 0.
    depth_max: float = 10.
    depth_scale: float = 1. / 5000.  # 16bit depth x depth_scale = floating value
    fx: float = 525.
    fy: float = 525.
    cx: float = 319.5
    cy: float = 293.5
    d0: float = 0.
    d1: float = 0.
    d2: float = 0.
    d3: float = 0.
    d4: float = 0.


class TumDataset(CameraBase):
    def __init__(self, dataset_dir: str, parameters: TumParameter | None = None):
        super().__init__()
        if parameters is None:
            parameters = TumParameter()
        self._parameters = parameters
        self._dataset_dir = dataset_dir
        self._file_number = self._find_image_files()
        self._file_count = 0
        self._prepare_data()

    def reset(self) -> None:
        self._file_count = 0

    def get_image(self) -> ImageWithFeature | None:
        if self._file_count >= self._file_number:
            return None
        image, color_image = self._read_image(self._image_files[self._file_count])
        depth = self._read_depth(self._depth_files[self._file_count])
        xyz = generate_xyz(self._x_over_z, self._y_over_z, depth)
        mask = self._make_mask_with_depth(
            depth, self._parameters.depth_min, self._parameters.depth_max)
        image_data = ImageWithFeature(image, color_image, xyz, mask)
        self._file_count += 1
        return image_data

    def _find_image_files(self) -> int:
        p = Path(self._dataset_dir)
        assert p.is_dir()
        self._image_files = [str(one_path) for one_path in list((p / 'rgb').glob('*.png'))]
        self._depth_files = [str(one_path) for one_path in list((p / 'depth').glob('*.png'))]
        self._image_files.sort()
        self._depth_files.sort()
        assert (self._image_files and self._depth_files)
        return min(len(self._image_files), len(self._depth_files))

    def _read_depth(self, filepath: str) -> NDArray:
        depth = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH) * self._parameters.depth_scale
        return depth

    def _read_image(self, filepath: str) -> tuple[NDArray, NDArray]:
        return read_image(filepath, self._camera_mat, self._distortion)

    def _prepare_data(self) -> None:
        # x = x_over_z * z, y = y_over_z * z
        self._x_over_z, self._y_over_z = prepare_generating_xyz(
            self._parameters.width,
            self._parameters.height,
            self._parameters.fx,
            self._parameters.fy,
            self._parameters.cx,
            self._parameters.cy)
        # camera matrix and distortion parameters
        self._camera_mat, self._distortion = camera_parameters_to_ndarray(
            self._parameters.fx,
            self._parameters.fy,
            self._parameters.cx,
            self._parameters.cy,
            self._parameters.d0,
            self._parameters.d1,
            self._parameters.d2,
            self._parameters.d3,
            self._parameters.d4)

    def _make_mask_with_depth(self, depth: NDArray, depth_min: float, depth_max: float) -> NDArray:
        is_over = depth > depth_min
        is_under = depth < depth_max
        mask = (is_over * is_under).astype(np.uint8)
        return mask
