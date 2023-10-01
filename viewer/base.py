
import numpy as np
import dearpygui.dearpygui as dpg
import platform
import time
from numpy.typing import NDArray
from matplotlib.pyplot import get_cmap
# local
from viewer.util import convert_to_dpg_image


# call it before generating tags
dpg.create_context()


IMAGE_TEXTURE_TAG = dpg.generate_uuid()
DEPTH_TEXTURE_TAG = dpg.generate_uuid()
TRAJECTORY_TAG = dpg.generate_uuid()
DRAW_LAYER_TAG = dpg.generate_uuid()
ORIENTATION_X_TAG = dpg.generate_uuid()
ORIENTATION_Y_TAG = dpg.generate_uuid()
ORIENTATION_Z_TAG = dpg.generate_uuid()


def make_empty_image(width: int, height: int, color: bool):
    shape = (height, width, 3) if color else (height, width)
    return np.zeros(shape, dtype=np.uint8)


class BaseViewer:
    def __init__(
            self, title: str, gui_width: int, gui_height: int, image_width: int, image_height: int,
            depth_width: int, depth_height: int, depth_min: float, depth_max: float,
            fps: float) -> None:
        self._gui_width = gui_width
        self._gui_height = gui_height
        self._image_width = image_width
        self._image_height = image_height
        self._depth_width = depth_width
        self._depth_height = depth_height
        self._sleep_time = 1. / fps
        self._is_macos = True if platform.system() == 'Darwin' else False
        # prepare gui
        self._depth_colors = self._generate_depth_color_map('rainbow')
        self.set_depth_min_max(depth_min, depth_max)
        self._setup_gui(title, self._gui_width, self._gui_height)
        self._setup_window()

    def start(self) -> None:
        dpg.show_viewport()

    def destroy(self) -> None:
        dpg.destroy_context()

    def is_running(self) -> bool:
        return dpg.is_dearpygui_running()

    def render(self) -> None:
        time.sleep(self._sleep_time)
        dpg.render_dearpygui_frame()

    def set_depth_min_max(self, min_: float, max_: float) -> None:
        self._depth_min = min_
        self._depth_max = max_

    def set_image(self, image: NDArray) -> None:
        gui_image = convert_to_dpg_image(image, self._is_macos)
        dpg.set_value(IMAGE_TEXTURE_TAG, gui_image)

    def set_depth(self, depth: NDArray) -> None:
        color_depth = self._colorize_depth(depth, self._depth_min, self._depth_max)
        gui_image = convert_to_dpg_image(color_depth, False)  # already has alpha channel
        dpg.set_value(DEPTH_TEXTURE_TAG, gui_image)

    def set_trajectory(self, trajectory: list[list[float]]) -> None:
        dpg.configure_item(TRAJECTORY_TAG, points=trajectory)

    def set_orientation(self, start: list[float], ends: list[list[float]]) -> None:
        for i, tag in enumerate((ORIENTATION_X_TAG, ORIENTATION_Y_TAG, ORIENTATION_Z_TAG)):
            dpg.configure_item(tag, p1=start)
            dpg.configure_item(tag, p2=ends[i])

    def _setup_gui(self, title: str, width: int, height: int) -> None:
        dpg.create_viewport(title=title, width=width, height=height)
        dpg.setup_dearpygui()

    def _setup_window(self) -> None:
        with dpg.texture_registry():
            # mvFormat_Float_rgb not currently supported on MacOS
            # https://dearpygui.readthedocs.io/en/latest/documentation/textures.html#formats
            _format = dpg.mvFormat_Float_rgba if self._is_macos else dpg.mvFormat_Float_rgb
            _image = make_empty_image(self._image_width, self._image_height, True)
            _depth = make_empty_image(self._depth_width, self._depth_height, True)
            dpg.add_raw_texture(
                self._image_width, self._image_height, convert_to_dpg_image(_image, self._is_macos),
                tag=IMAGE_TEXTURE_TAG, format=_format,
                use_internal_label=False)
            dpg.add_raw_texture(
                self._depth_width, self._depth_height, convert_to_dpg_image(_depth, self._is_macos),
                tag=DEPTH_TEXTURE_TAG, format=_format,
                use_internal_label=False)
        # color image
        with dpg.window(label='Image', pos=(0, 0)):
            dpg.add_image(IMAGE_TEXTURE_TAG, width=self._image_width, height=self._image_height)
        # depth image
        with dpg.window(label='Depth', pos=(self._image_width + 20, 0)):
            dpg.add_image(DEPTH_TEXTURE_TAG, width=self._depth_width, height=self._depth_height)
        # 3d Trajectory
        h_space = 20
        _height = self._gui_height - self._image_height - h_space
        with dpg.window(label='Trajectory', pos=(0, self._image_height + h_space)):
            with dpg.drawlist(
                    width=self._gui_width, height=_height):
                self._make_trajectory_window()
        dpg.set_clip_space(DRAW_LAYER_TAG, 0, 0, self._gui_width, _height, -5.0, 5.0)


    def _generate_depth_color_map(self, name: str = 'rainbow') -> NDArray:
        cmap = get_cmap(name)
        colors = (np.array([cmap(i / 255) for i in range(256)]) * 255).astype(np.uint8)
        # smallest value -> black
        colors = colors[::-1, ...]
        colors[0, :3] = 0
        return colors

    def _colorize_depth(self, depth: NDArray, min_: float, max_: float) -> NDArray:
        tmp = np.clip(depth, min_, max_)
        tmp = ((tmp - min_) / (max_ - min_) * 255).astype(np.uint8)
        rgb = self._depth_colors[tmp]
        return rgb

    def _make_trajectory_window(self) -> None:
        with dpg.draw_layer(
                tag=DRAW_LAYER_TAG, depth_clipping=True, perspective_divide=True,
                cull_mode=dpg.mvCullMode_Back):
            dpg.draw_polyline([[]], tag=TRAJECTORY_TAG, thickness=1, closed=False, show=True)
            dpg.draw_line([], [], tag=ORIENTATION_X_TAG, color=(255, 0, 0))
            dpg.draw_line([], [], tag=ORIENTATION_Y_TAG, color=(0, 255, 0))
            dpg.draw_line([], [], tag=ORIENTATION_Z_TAG, color=(4, 207, 228))
