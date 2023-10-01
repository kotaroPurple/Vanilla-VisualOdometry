
import numpy as np
from numpy.typing import NDArray
# local
from vo.functions import inverse_transformation_matrix


class Trajectory:
    def __init__(self, store_number: int) -> None:
        self._max_number = store_number
        self._orientation = np.eye(3)
        self._position_list: list[list[float]] = []

    def append_pose(self, t_mat: NDArray) -> None:
        # append position
        if len(self._position_list) >= self._max_number:
            self._position_list.pop(0)
        self._position_list.append(t_mat[:-1, -1].tolist())
        # current orientation
        inv_t_mat = inverse_transformation_matrix(t_mat)
        self._orientation = inv_t_mat[:-1, :-1]

    def get_trajectory(self) -> list[list[float]]:
        return self._position_list

    def get_position(self) -> list[float]:
        return self._position_list[-1]

    def get_orientation(self) -> list[list[float]]:
        return self._orientation.tolist()

    def get_orientation_end(self, length: float) -> list[list[float]]:
        tranlation = np.array(self.get_position())
        end_points = tranlation + length * self._orientation
        return end_points.tolist()
