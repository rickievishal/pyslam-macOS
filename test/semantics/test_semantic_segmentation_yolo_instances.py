from types import SimpleNamespace

import numpy as np
import torch

from pyslam.semantics.semantic_segmentation_yolo import SemanticSegmentationYolo
from pyslam.semantics.semantic_types import SemanticFeatureType


def test_yolo_label_output_keeps_instance_ids_and_reserves_background():
    segmenter = SemanticSegmentationYolo.__new__(SemanticSegmentationYolo)
    segmenter.semantic_feature_type = SemanticFeatureType.LABEL
    segmenter.class_id_offset = 1
    segmenter.num_classes_with_background = 81

    masks = torch.zeros((2, 4, 4), dtype=torch.float32)
    masks[0, :3, :3] = 1
    masks[1, 1:, 1:] = 1
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(
            cls=torch.tensor([0, 2]),
            conf=torch.tensor([0.9, 0.8]),
        ),
    )

    semantics, instances = segmenter._results_to_semantics(result, (4, 4))

    assert semantics.dtype == np.int32
    assert instances.dtype == np.int32
    assert semantics[0, 0] == 1
    assert semantics[3, 3] == 3
    assert instances[0, 0] == 1
    assert instances[3, 3] == 2
    assert np.all(semantics[instances == 0] == 0)
