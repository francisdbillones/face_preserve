import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

NON_ROI_CRF = 45
ROI_CRF = 30
DRAWBOX = False
CODEC = "libx264"
PRESET = "veryfast"


BLAZEFACE_UPDATE_FREQ = 4
BLAZEFACE_PATH = "detectors/blazeface.tflite"
BLAZEFACE_OPTIONS = mp_vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=BLAZEFACE_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
)


HAARCASCADE_UPDATE_FREQ = 8

HAARCASCADE_FACE_PATH = "detectors/face_cascade.xml"
HAARCASCADE_FACE_OPTIONS = {
    "scaleFactor": 1.1,
    "minNeighbors": 5,
    "minSize": (25, 25),
}

HAARCASCADE_FULLBODY_PATH = "detectors/fullbody_cascade.xml"
HAARCASCADE_FULLBODY_OPTIONS = {
    "scaleFactor": 1.01,
    "minNeighbors": 5,
    "minSize": (25, 25),
}

HOG_UPDATE_FREQ = 8
HOG_FULLBODY_OPTIONS = {
    "winStride": (4, 4),
    "padding": (0, 0),
    "scale": 1.05,
}
