import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

NON_ROI_CRF = 45
ROI_CRF = 30
DRAWBOX = False
CODEC = "libx264"
PRESET = "veryfast"
UPDATE_FREQ = 4


BLAZEFACE_PATH = "detectors/blazeface.tflite"
BLAZEFACE_OPTIONS = mp_vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=BLAZEFACE_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
)
