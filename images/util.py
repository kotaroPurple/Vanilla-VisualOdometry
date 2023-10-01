
import numpy as np
import cv2
from numpy.typing import NDArray


def read_image(
        filepath: str, camera_mat: NDArray | None = None, distortion: NDArray | None = None) \
        -> tuple[NDArray, NDArray]:
    img = cv2.imread(filepath)
    if (camera_mat is not None) and (distortion is not None):
        if not np.allclose(distortion, np.zeros_like(distortion)):
            img = cv2.undistort(img, camera_mat, distortion)
    # grascale
    if img.ndim == 2:
        return img, np.array([])
    # color -> grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image, img


def read_image_as_gray(filepath: str) -> NDArray:
    img = cv2.imread(filepath)
    # grascale
    if img.ndim == 2:
        return img
    # color -> grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image


def grayscale_to_3ch(image: NDArray) -> NDArray:
    return np.tile(image[:, :, np.newaxis], (1, 1, 3))


def opencv_color_to_rgb(image: NDArray) -> NDArray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_opencv_color(image: NDArray) -> NDArray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ndarray_from_keypoints(keypoints: tuple[cv2.KeyPoint]) -> NDArray:
    return cv2.KeyPoint_convert(keypoints)


def keypoints_from_ndarray(keypoints: NDArray) -> list[cv2.KeyPoint]:
    if len(keypoints) == 0:
        return []
    size, angle, response, octave, class_id = 15, 0., 0, 0, -1
    result = [cv2.KeyPoint(x, y, size, angle, response, octave, class_id) for (x, y) in keypoints]
    return result


def draw_keypoints_on_image(
        image: NDArray, keypoints: tuple[cv2.KeyPoint] | NDArray, is_bgr: bool = True) -> NDArray:
    # convert NDArray to keypoints
    if isinstance(keypoints, np.ndarray):
        keypoints = keypoints_from_ndarray(keypoints)
    if is_bgr is False:
        image = rgb_to_opencv_color(image)
    red = (0, 0, 255)
    result = cv2.drawKeypoints(
        image, keypoints, 0, red, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return result
