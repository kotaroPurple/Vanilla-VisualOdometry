
import numpy as np
from numpy.typing import NDArray


def add_alpha_channel(image: NDArray, alpha: int = 255) -> NDArray:
    rgba = np.dstack((image, np.full(image.shape[:2], alpha, dtype=image.dtype)))
    return rgba


def convert_to_dpg_image(image: NDArray, is_mac: bool) -> NDArray:
    if is_mac:
        image = add_alpha_channel(image, 255)
    dpg_image = np.true_divide(image.astype(np.float32), 255).ravel()
    return dpg_image
