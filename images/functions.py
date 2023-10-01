
import cv2
from numpy.typing import NDArray


def remove_outliers_by_fundamental_matrix(
        points1: NDArray, points2: NDArray, ransac_threshold: float) -> tuple[NDArray, NDArray]:
    _, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, ransac_threshold)
    good_points = mask.ravel() == 1
    inliers1 = points1[good_points]
    inliers2 = points2[good_points]
    return inliers1, inliers2
