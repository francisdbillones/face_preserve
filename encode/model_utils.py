from typing import List, NamedTuple, Tuple, Union
import math
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class BBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    if not (
        _is_valid_normalized_value(normalized_x)
        and _is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def _is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (
        value < 1 or math.isclose(1, value)
    )


def iou(box1: BBox, box2: BBox) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    intersection = max(0, xB - xA) * max(0, yB - yA)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection

    iou = intersection / union

    assert 0 <= iou <= 1

    return iou


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


class HaarCascadeDetector:
    def __init__(self, haar_cascade_filename: str, options: dict):
        self._haar_cascade_detector = cv2.CascadeClassifier(haar_cascade_filename)
        self._options = options

    def detect(self, image: np.ndarray) -> List[BBox]:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("yipee")
        haar_cascade_bboxes = [
            BBox(x=x, y=y, w=w, h=h)
            for (
                x,
                y,
                w,
                h,
            ) in self._haar_cascade_detector.detectMultiScale(image, **self._options)
        ]
        return haar_cascade_bboxes


class HOGDetector:
    def __init__(self, hog_descriptor: cv2.HOGDescriptor, options: dict):
        self._hog_descriptor = hog_descriptor
        self._options = options

    def detect(self, image: np.ndarray) -> List[BBox]:
        hog_bboxes = [
            BBox(x=x, y=y, w=w, h=h)
            for (
                x,
                y,
                w,
                h,
            ) in self._hog_descriptor.detectMultiScale(
                image, **self._options
            )[0]
        ]
        return hog_bboxes


def load_model(filename: str) -> mp_vision.FaceDetector:
    blazeface_base_options = mp_python.BaseOptions(model_asset_path=filename)
    blazeface_options = mp_vision.FaceDetectorOptions(
        base_options=blazeface_base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    detector = BlazeFaceDetector(blazeface_options)

    return detector
