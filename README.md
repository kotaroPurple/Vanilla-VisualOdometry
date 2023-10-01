# Vanilla-VisualOdometry

I created this project as a study exercise to implement a simple Visual Odometry system in Python.
Please note that this project is currently in development.

This project is aimed at devices that can capture camera images and depth images, such as those using the TUM Dataset or Oak-D.

## How to Use
1. Clone this repository
2. Download TUM Dataset (Sequence 'freiburg1_xyz')
3. Run run_tumdataset.py

```bash
python run_tumdataset.py --dataset /path/to/rgbd_dataset_freiburg1_xyz/
```

## Result
Not good result.
![Not good result](./gif/trajectory_tumdataset.gif)

## TODO
- Bug fixes
- Integration with Oak-D
- Implement pose filtering (https://github.com/kotaroPurple/lie_solver)
- Etc.
