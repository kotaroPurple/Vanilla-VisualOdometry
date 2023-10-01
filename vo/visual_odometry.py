
import numpy as np
from numpy.typing import NDArray
# local
from vo.functions import calculate_transformation_between_points
from vo.functions import make_identity_matrix
from vo.functions import inverse_transformation_matrix
from images.functions import remove_outliers_by_fundamental_matrix
from images.image_class import FeatureDetector
from images.image_class import ImageWithFeature
from images.util import ndarray_from_keypoints


class VisualOdometry:
    def __init__(self, outlier_threshold: float = 3.):
        self._previous_image: ImageWithFeature = None
        self._current_image: ImageWithFeature = None
        self._detector = FeatureDetector()
        self._outlier_threshold = outlier_threshold
        self.set_pose(make_identity_matrix())
        self._move = make_identity_matrix()
        self._inlier_keypoints = np.array([])

    def set_pose(self, pose: NDArray) -> None:
        self._current_pose = pose

    def get_pose(self) -> NDArray:
        return self._current_pose

    def get_move(self) -> NDArray:
        return self._move

    def step_backforward(self) -> None:
        self._current_image = self._previous_image
        self._update_pose(inverse_transformation_matrix(self._move))

    def get_keypoints(self) -> NDArray:
        return self._inlier_keypoints

    def calculate(self, new_image: ImageWithFeature) -> NDArray:
        # current -> previous
        self._previous_image = self._current_image
        self._current_image = new_image
        # detect features from the new image
        self._current_image.detect_features(self._detector)
        # exit if no pair exist
        if self._previous_image is None:
            return make_identity_matrix()
        # feature matching
        matches = self._detector.match(
            self._previous_image.get_descriptors(), self._current_image.get_descriptors())
        # feature associaiton
        previous_keypoints = self._previous_image.get_keypoints()
        current_keypoints = self._current_image.get_keypoints()
        previous_matched_keypoints = [previous_keypoints[match.queryIdx] for match in matches]
        current_matched_keypoints = [current_keypoints[match.trainIdx] for match in matches]
        # remove outliers with Fundamental Matrix (RANSAC)
        previous_inlier, current_inlier = remove_outliers_by_fundamental_matrix(
            ndarray_from_keypoints(previous_matched_keypoints),
            ndarray_from_keypoints(current_matched_keypoints),
            self._outlier_threshold)
        # predict transformation matrix between feature points
        prev_xyz = self._previous_image.get_key_xyz(previous_inlier)
        current_xyz = self._current_image.get_key_xyz(current_inlier)
        self._move = calculate_transformation_between_points(prev_xyz, current_xyz)
        self._update_pose(self._move)
        # store current keypoints
        self._inlier_keypoints = current_inlier
        return self._move

    def _update_pose(self, move: NDArray) -> None:
        # T1.r1 = r0, T2.r2 = r1 -> r0 = T1.T2.r2
        self._current_pose = self._current_pose @ move
