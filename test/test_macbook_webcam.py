import numpy as np
from types import SimpleNamespace

from pyslam.io.dataset import LiveDataset
from pyslam.io import dataset_factory
from pyslam.local_features.feature_manager import feature_manager_factory
from pyslam.local_features.feature_types import FeatureDescriptorTypes, FeatureDetectorTypes


class FakeCapture:
    def __init__(self, opened=True, frame=None):
        self.opened = opened
        self.frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        self.released = False
        self.properties = {}

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.properties[key] = value
        return True

    def get(self, key):
        return {3: 640, 4: 480, 5: 30}.get(key, 0)

    def read(self):
        return True, self.frame

    def release(self):
        self.released = True


def test_live_dataset_configures_and_releases_camera(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr("pyslam.io.dataset.cv2.VideoCapture", lambda _: capture)
    dataset = LiveDataset("", 0, width=640, height=480, fps=30)
    frame = dataset.getImageColor(0)
    assert frame.shape == (480, 640, 3)
    assert dataset.num_frames is None
    assert dataset.is_ok
    assert dataset.getTimestamp() is not None
    dataset.quit()
    assert capture.released


def test_live_dataset_rejects_unavailable_camera(monkeypatch):
    monkeypatch.setattr("pyslam.io.dataset.cv2.VideoCapture", lambda _: FakeCapture(opened=False))
    try:
        LiveDataset("", 7)
    except IOError as error:
        assert "Cannot open camera 7" in str(error)
    else:
        raise AssertionError("Expected unavailable camera to raise IOError")


def test_live_factory_uses_environment_overrides(monkeypatch):
    captured = {}

    def fake_live_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    class Config:
        dataset_settings = {
            "type": "live", "name": 0, "base_path": "", "sensor_type": "mono",
            "width": 640, "height": 480, "fps": 30, "environment_type": "indoor",
        }
        cam_settings = {}
        cam_stereo_settings = None

    monkeypatch.setattr(dataset_factory, "LiveDataset", fake_live_dataset)
    monkeypatch.setenv("CAMERA_ID", "2")
    monkeypatch.setenv("WIDTH", "800")
    monkeypatch.setenv("HEIGHT", "600")
    monkeypatch.setenv("FPS", "24")
    dataset_factory.dataset_factory(Config())
    assert captured["args"][1] == "2"
    assert captured["kwargs"] == {
        "width": 800,
        "height": 600,
        "fps": 24.0,
        "environment_type": dataset_factory.DatasetEnvironmentType.INDOOR,
    }


def test_orb_feature_manager_extracts_features():
    manager = feature_manager_factory(
        num_features=500,
        num_levels=8,
        detector_type=FeatureDetectorTypes.ORB,
        descriptor_type=FeatureDescriptorTypes.ORB,
    )
    image = np.random.default_rng(0).integers(0, 256, (480, 640), dtype=np.uint8)

    keypoints, descriptors = manager.detectAndCompute(image)
    assert len(keypoints) > 0
    assert len(keypoints) == len(descriptors)
