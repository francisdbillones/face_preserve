import ffmpeg
from model_utils import BBox

from typing import List, NamedTuple


class VideoInfo(NamedTuple):
    width: int
    height: int
    duration_s: float
    fps: int


def read_bboxes(filename: str):
    with open(filename) as f:
        bboxes = eval(f.read())
    return bboxes


def add_rois(
    stream: ffmpeg.nodes.FilterableStream,
    frame_bboxes: List[BBox],
    qoffset: float,
    drawbox=False,
):
    for bbox in frame_bboxes:
        stream = stream.filter(
            "addroi", x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h, qoffset=qoffset
        )
        if drawbox:
            stream = stream.drawbox(
                x=bbox.x, y=bbox.y, width=bbox.w, height=bbox.h, color="red"
            )
    return stream


def get_video_info(filename: str):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width = int(video_info["width"])
    height = int(video_info["height"])
    duration_s = float(probe["format"]["duration"])
    fps = eval(video_info["r_frame_rate"])
    return VideoInfo(width, height, duration_s, fps)
