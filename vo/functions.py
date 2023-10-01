
import numpy as np
from numpy.typing import NDArray


def decompose_transformation_matrix(t_mat: NDArray) -> tuple[NDArray, NDArray]:
    return t_mat[:-1, :-1], t_mat[:-1, -1]


def calculate_transformation_between_points(
        previous_points: NDArray, current_points: NDArray) -> NDArray:
    # calculate T (holds T.Current = Previous)
    # ref: https://www.jstage.jst.go.jp/article/jrsj/31/6/31_31_624/_pdf
    p_mean = np.mean(previous_points, axis=0)
    c_mean = np.mean(current_points, axis=0)
    p_minus_mean = previous_points - p_mean
    c_minus_mean = current_points - c_mean
    rot_mat = calculate_rotation_between_points(p_minus_mean, c_minus_mean)
    translation = p_mean - rot_mat @ c_mean
    n_dim = previous_points.shape[1]
    t_mat = np.eye(n_dim + 1)
    t_mat[:-1, :-1] = rot_mat
    t_mat[:-1, -1] = translation
    return t_mat


def calculate_rotation_between_points(previous_points: NDArray, current_points: NDArray) -> NDArray:
    # calculate R (holds R.Current = Previous)
    mat_x = previous_points.T @ current_points
    mat_u, _, mat_vt = np.linalg.svd(mat_x, full_matrices=True)
    mat_s = np.diag([1., 1., np.linalg.det(mat_u @ mat_vt)])
    rot_mat = mat_u @ mat_s @ mat_vt
    return rot_mat


def calculate_rotation_without_association(
        previous_points: NDArray, current_points: NDArray) -> tuple[NDArray, NDArray]:
    # calculate R (holds R.Current = Previous)
    # ref: https://www.jstage.jst.go.jp/article/jrsj/31/6/31_31_624/_pdf
    A = current_points
    B = previous_points
    A_u, _, A_vh = np.linalg.svd(A, full_matrices=True)
    B_u, _, B_vh = np.linalg.svd(B, full_matrices=True)
    rot = B_vh.T @ A_vh
    P = A_u.T @ B_u
    return rot, P


def inverse_transformation_matrix(t_mat: NDArray) -> NDArray:
    # (R, t).inv = (R.T, -R.T*t)
    # (0, 1)       (0, 1)
    inv_mat = np.zeros_like(t_mat)
    rot_t = t_mat[:-1, :-1].T
    translation = t_mat[:-1, -1]
    inv_mat[:-1, :-1] = rot_t
    inv_mat[:-1, -1] = -rot_t @ translation
    inv_mat[-1, -1] = 1.
    return inv_mat


def make_identity_matrix() -> NDArray:
    return np.eye(4)


def angle_from_rotation_matrix(rot_mat: NDArray) -> float:
    # trace(R) = 2.cos(theta) + 1
    # theta = arccos((trace(R) - 1) / 2)
    _trace = np.trace(rot_mat)
    return np.arccos(0.5 * (_trace - 1.))


def length_from_translation(translation: NDArray) -> float:
    return np.sqrt(translation.dot(translation))


def angle_and_length_from_transformation(t_mat: NDArray) -> tuple[float, float]:
    rot, trans = decompose_transformation_matrix(t_mat)
    angle = angle_from_rotation_matrix(rot)
    length = length_from_translation(trans)
    return angle, length
