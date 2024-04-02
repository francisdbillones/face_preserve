from typing import List, NamedTuple
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

from constants import BLAZEFACE_OPTIONS


class BBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


def GET_MODEL():
    # modify this to return the model you want to use
    return BlazeFaceDetector(BLAZEFACE_OPTIONS)


class BlazeFaceDetector:
    def __init__(
        self,
        blazeface_options: mp_vision.FaceDetectorOptions,
    ):
        self._options = blazeface_options
        self._blazeface_face_detector = mp_vision.FaceDetector.create_from_options(
            self._options
        )

    def detect(self, image: np.ndarray, frame_timestamp_ms: int) -> List[BBox]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        blazeface_bboxes = [
            BBox(
                x=detection.bounding_box.origin_x,
                y=detection.bounding_box.origin_y,
                w=detection.bounding_box.width,
                h=detection.bounding_box.height,
            )
            for detection in self._blazeface_face_detector.detect_for_video(
                mp_image, frame_timestamp_ms
            ).detections
        ]
        return blazeface_bboxes
