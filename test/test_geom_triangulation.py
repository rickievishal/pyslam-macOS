import numpy as np

from pyslam.utilities import geom_triangulation


def test_triangulate_normalized_points_returns_per_point_validity_mask(monkeypatch):
    homogeneous_points = np.array(
        [
            [2.0, 4.0, 6.0],
            [4.0, 8.0, 12.0],
            [6.0, 12.0, 18.0],
            [2.0, 0.0, 3.0],
        ]
    )
    monkeypatch.setattr(
        geom_triangulation.cv2,
        "triangulatePoints",
        lambda *_: homogeneous_points,
    )
    poses = np.eye(4)
    keypoints = np.zeros((3, 2))

    points_3d, valid_mask = geom_triangulation.triangulate_normalized_points(
        poses, poses, keypoints, keypoints
    )

    assert points_3d.shape == (3, 3)
    assert valid_mask.shape == (3,)
    assert valid_mask.dtype == np.bool_
    np.testing.assert_array_equal(valid_mask, [True, False, True])
    np.testing.assert_allclose(
        points_3d,
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [2.0, 4.0, 6.0]],
    )
