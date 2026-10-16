from threading import RLock

import numpy as np

from pyslam.slam.map_point import MapPointBase


class FakeFrame:
    is_keyframe = False

    def __init__(self, num_keypoints=1):
        self.points = [None] * num_keypoints

    def get_point_match(self, idx):
        return self.points[idx]

    def set_point_match(self, point, idx):
        self.points[idx] = point

    def remove_point_match(self, idx):
        self.points[idx] = None

    def remove_point(self, point):
        self.points = [None if match == point else match for match in self.points]


class FakeKeyFrame:
    is_keyframe = True

    def __init__(self):
        self.points = np.array([None], dtype=object)
        self.kps_ur = np.array([-1.0])
        self._lock_features = RLock()

    def get_point_match(self, idx):
        return self.points[idx]

    def set_point_match(self, point, idx):
        self.points[idx] = point

    def remove_point_match(self, idx):
        self.points[idx] = None

    def remove_point(self, point):
        self.points[self.points == point] = None


def test_remove_frame_view_clears_an_existing_frame_association():
    point = MapPointBase()
    frame = FakeFrame()

    assert point.add_frame_view(frame, 0)
    point.remove_frame_view(frame, 0)

    assert frame.get_point_match(0) is None
    assert point.get_frame_view_idx(frame) == -1


def test_remove_keyframe_observation_clears_its_single_valid_match():
    point = MapPointBase()
    keyframes = [FakeKeyFrame() for _ in range(4)]

    for keyframe in keyframes:
        assert point.add_observation(keyframe, 0)

    point.remove_observation(keyframes[0], 0, map_no_lock=True)

    assert keyframes[0].get_point_match(0) is None
    assert point.get_observation_idx(keyframes[0]) == -1
    assert all(keyframe.get_point_match(0) is point for keyframe in keyframes[1:])
    assert point.num_observations() == 3


def test_remove_frame_view_handles_a_frame_slot_cleared_by_outlier_cleanup():
    point = MapPointBase()
    frame = FakeFrame()

    assert point.add_frame_view(frame, 0)
    frame.remove_point_match(0)
    point.remove_frame_view(frame, 0)

    assert frame.get_point_match(0) is None
    assert point.get_frame_view_idx(frame) == -1
