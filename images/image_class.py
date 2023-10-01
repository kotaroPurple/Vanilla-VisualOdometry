
import numpy as np
import cv2
from numpy.typing import NDArray
from dataclasses import dataclass
# local
from images.util import ndarray_from_keypoints


def _float_points_to_int(points: NDArray) -> NDArray:
    return (points + 0.5).astype(np.int32)


# ORB Parameters
# https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html?highlight=cv2.orb#cv2.ORB
@dataclass
class OrbParameter:
    nfeatures: int = 500
    scaleFactor: float = 1.2
    nlevels: int = 8
    edgeThreshold: int = 31
    firstLevel: int = 0
    WTA_K: int = 2
    scoreType: int = cv2.ORB_HARRIS_SCORE
    patchSize: int = 31


# add orb parameters
class FeatureDetector:
    def __init__(self, orb_param: OrbParameter | None = None):
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if orb_param is None:
            orb_param = OrbParameter()
        self._detector = cv2.ORB_create(
            nfeatures=orb_param.nfeatures,
            scaleFactor=orb_param.scaleFactor,
            nlevels=orb_param.nlevels,
            edgeThreshold=orb_param.edgeThreshold,
            firstLevel=orb_param.firstLevel,
            WTA_K=orb_param.WTA_K,
            scoreType=orb_param.scoreType,
            patchSize=orb_param.patchSize)

    def detect(self, image: NDArray, mask: NDArray | None = None) \
            -> tuple[tuple[cv2.KeyPoint], NDArray]:
        keypoints, descriptors = self._detector.detectAndCompute(image, mask)
        return keypoints, descriptors

    def match(self, descriptor1: NDArray, descriptor2: NDArray) -> tuple[cv2.DMatch]:
        matches = self._matcher.match(descriptor1, descriptor2)
        return matches


class ImageWithFeature:
    def __init__(
            self, image: NDArray | None = None, color_image: NDArray | None = None,
            xyz: NDArray | None = None, mask: NDArray | None = None):
        self.set_image(image)
        self.set_color_image(color_image)
        self.set_xyz(xyz)
        self.set_mask(mask)
        self._keypoints = None
        self._descriptors = None
        self._np_keypoints = None
        self._key_xyz = None

    def set_image(self, image: NDArray | None) -> None:
        self._image = image

    def set_color_image(self, image: NDArray | None) -> None:
        self._color_image = image

    def set_xyz(self, xyz: NDArray | None) -> None:
        self._xyz = xyz

    def set_mask(self, mask: NDArray | None) -> None:
        self._mask = mask

    def get_image(self) -> NDArray:
        return self._image

    def get_color_image(self, reverse_color: bool = False) -> NDArray | None:
        if reverse_color:
            return self._color_image[..., ::-1]
        else:
            return self._color_image

    def get_xyz(self) -> NDArray:
        return self._xyz

    def get_depth(self) -> NDArray:
        return self._xyz[..., 2]

    def get_x_depth(self) -> NDArray:
        return self._xyz[..., 0]

    def get_y_depth(self) -> NDArray:
        return self._xyz[..., 1]

    def get_mask(self) -> NDArray:
        return self._mask

    def detect_features(self, detector: FeatureDetector) -> None:
        if self._image is not None:
            self._keypoints, self._descriptors = detector.detect(self._image, self._mask)
            self._generate_xyz_at_keypoints()
        else:
            self._keypoints, self._descriptors = None, None

    def _generate_xyz_at_keypoints(self) -> None:
        if (self._xyz is not None) and (self._keypoints is not None):
            points = self.get_ndarray_keypoints(as_int=True)
            self._key_xyz = self._xyz[points[:, 1], points[:, 0]]

    def get_keypoints(self) -> tuple[cv2.KeyPoint] | None:
        return self._keypoints

    def get_ndarray_keypoints(self, as_int: bool = True) -> NDArray | None:
        if self._np_keypoints is None:
            self._np_keypoints = ndarray_from_keypoints(self._keypoints)
            if as_int:
                self._np_keypoints = _float_points_to_int(self._np_keypoints)
        return self._np_keypoints

    def get_descriptors(self) -> NDArray:
        return self._descriptors

    def get_key_xyz(self, points: NDArray) -> NDArray:
        if points.dtype in (np.float32, np.float64):
            points = _float_points_to_int(points)
        return self._xyz[points[:, 1], points[:, 0]]
