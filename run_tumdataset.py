
import argparse
from viewer.base import BaseViewer
from camera.tum_dataset import TumDataset
from camera.tum_dataset import TumParameter
from vo.visual_odometry import VisualOdometry
from vo.trajectory import Trajectory
from images.util import draw_keypoints_on_image


def main(args):
    # GUI
    img_width = 640
    img_height = 480
    depth_width = 640
    depth_height = 480
    gui = BaseViewer(
        'Visual Odometry',
        1400, 1000, img_width, img_height, depth_width, depth_height, 0., 5., 30.)
    gui.start()

    # prepare Visual Odometry
    tum_dir = args.dataset
    tum_param = TumParameter()
    data_loader = TumDataset(tum_dir, tum_param)
    vo = VisualOdometry(outlier_threshold=args.threshold)
    traj = Trajectory(100)

    # run Visual Odometry
    while gui.is_running():
        # read image
        img = data_loader.get_image()
        if img is None:
            break
        # odometry
        vo.calculate(img)
        # get images
        color_image = img.get_color_image()
        z_image = img.get_depth()
        # draw keypoints
        keypoints = vo.get_keypoints()
        key_image = draw_keypoints_on_image(color_image, keypoints, False)
        # show images
        gui.set_image(key_image)
        gui.set_depth(z_image)
        # set trajectory
        traj.append_pose(vo.get_pose())
        position_list = traj.get_trajectory()
        current_position = traj.get_position()
        ori_end_points = traj.get_orientation_end(length=0.03)
        gui.set_trajectory(position_list)
        gui.set_orientation(current_position, ori_end_points)
        # render
        gui.render()
    gui.destroy()


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--threshold', type=float, default=3.0)
    args = parser.parse_args()
    # run main function
    main(args)
