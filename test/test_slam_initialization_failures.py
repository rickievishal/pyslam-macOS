from types import SimpleNamespace

from pyslam.io.dataset_types import SensorType
from pyslam.slam.initializer import Initializer
from pyslam.slam.slam import Slam


def test_rgbd_initializer_uses_depth_initialization_feature_threshold():
    initializer = Initializer(SensorType.RGBD)

    assert initializer.num_min_features == 100
    assert initializer.num_min_triangulated_points == 100


def test_stereo_initializer_keeps_stereo_feature_threshold():
    initializer = Initializer(SensorType.STEREO)

    assert initializer.num_min_features == 500


def test_final_trajectory_is_empty_when_slam_has_no_keyframes():
    slam = SimpleNamespace(map=SimpleNamespace(get_keyframes=lambda: []))

    assert Slam.get_final_trajectory(slam) == ([], [], [])
