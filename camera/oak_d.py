
import cv2
import depthai as dai
import numpy as np
from numpy.typing import NDArray
# # local
# from camera.base import CameraBase
# from camera.util import prepare_generating_xyz
# from camera.util import generate_xyz
# from camera.util import camera_parameters_to_ndarray
# from images.util import read_image
# from images.image_class import ImageWithFeature


class OakD:
    def __init__(self):
        self._device = dai.Device()
        self._queue_names = ["rgb", "disp"]
        self._fps = 30
        self._pipeline = self._create_pipeline()

    def start_pipeline(self) -> None:
        self._device.startPipeline(self._pipeline)

    def stop_pipeline(self) -> None:
        self._device.close()

    def _create_pipeline(self) -> None:
        pipeline = dai.Pipeline()
        # inputs
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        left = pipeline.create(dai.node.MonoCamera)
        right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        # properties
        # # rgb
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        cam_rgb.setFps(self._fps)
        # # left
        mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        left.setResolution(mono_resolution)
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left.setFps(self._fps)
        # # right
        right.setResolution(mono_resolution)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right.setFps(self._fps)
        # # stereo
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # outpus
        rgb_out = pipeline.create(dai.node.XLinkOut)
        disparity_out = pipeline.create(dai.node.XLinkOut)
        # set streams
        rgb_out.setStreamName(self._queue_names[0])  # rgb
        disparity_out.setStreamName(self._queue_names[1])  # depth
        # linking
        cam_rgb.isp.link(rgb_out.input)
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.disparity.link(disparity_out.input)
        self.stereo = stereo
        # output
        return pipeline

    def get_image(self):
        pass

    def get_data(self):
        latest_packet = {}
        queue_events = self._device.getQueueEvents(self._queue_names)
        for queue_name in queue_events:
            packets = self._device.getOutputQueue(queue_name).tryGetAll()
            if len(packets) > 0:
                latest_packet[queue_name] = packets[-1]
        frame_rgb = None
        frame_disp = None
        if "rgb" in latest_packet:
            frame_rgb = latest_packet["rgb"].getCvFrame()

        if "disp" in latest_packet:
            frame_disp = latest_packet["disp"].getFrame()
            max_disparity = self.stereo.initialConfig.getMaxDisparity()
            frame_disp = (frame_disp * 255.0 / max_disparity).astype(np.uint8)
            frame_disp = cv2.applyColorMap(frame_disp, cv2.COLORMAP_HOT)
        return frame_rgb, frame_disp

    def test_run(self):
        with self._device:
            self._device.startPipeline(self._pipeline)

            cv2.namedWindow("rgb")
            cv2.namedWindow("disp")

            while True:
                latest_packet = {}

                queue_events = self._device.getQueueEvents(self._queue_names)
                for queue_name in queue_events:
                    packets = self._device.getOutputQueue(queue_name).tryGetAll()
                    if len(packets) > 0:
                        latest_packet[queue_name] = packets[-1]

                if "rgb" in latest_packet:
                    frame_rgb = latest_packet["rgb"].getCvFrame()
                    cv2.imshow("rgb", frame_rgb)

                if "disp" in latest_packet:
                    frame_disp = latest_packet["disp"].getFrame()
                    max_disparity = self.stereo.initialConfig.getMaxDisparity()
                    if 1:
                        frame_disp = (frame_disp * 255.0 / max_disparity).astype(np.uint8)
                    if 1:
                        frame_disp = cv2.applyColorMap(frame_disp, cv2.COLORMAP_HOT)
                    frame_disp = np.ascontiguousarray(frame_disp)
                    cv2.imshow("disp", frame_disp)

                if cv2.waitKey(1) == ord("q"):
                    break


if __name__ == "__main__":
    oak_d = OakD()
    oak_d.start_pipeline()
    while True:
        cv2.namedWindow("rgb")
        cv2.namedWindow("disp")
        rgb_image, disparity_map = oak_d.get_data()
        if (rgb_image is not None) and (disparity_map is not None):
            cv2.imshow("rgb", rgb_image)
            cv2.imshow("disp", disparity_map)
        if cv2.waitKey(1) == ord("q"):
            break
    # end of process
    oak_d.stop_pipeline()
    cv2.destroyAllWindows()
