import subprocess

import ffmpeg
import numpy as np

from model_utils import BBox
from constants import CODEC, PRESET

from typing import List, NamedTuple


class VideoInfo(NamedTuple):
    width: int
    height: int
    duration_s: float
    fps: int


def ffmpeg_video_to_rgb24_process(in_filename, pipe=subprocess.PIPE):
    args = (
        ffmpeg.input(in_filename)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .compile()
    )
    return subprocess.Popen(args, stdout=pipe)


def ffmpeg_add_bboxes_process(
    in_filename: str,
    out_filename: str,
    base_crf: int,
    roi_crf: int,
    bboxes: List[BBox],
    update_freq: int,
    in_video_info: VideoInfo,
    drawbox: bool = False,
):
    true_video_fps = in_video_info.fps
    duration_s = in_video_info.duration_s
    in_file = ffmpeg.input(in_filename)

    args = (
        ffmpeg.concat(
            *(
                add_rois(
                    in_file.trim(  # chunk
                        start=(i * update_freq / true_video_fps),
                        end=(i + 1) * update_freq / true_video_fps,
                    ).filter("setpts", "PTS-STARTPTS"),
                    bboxes[(min(len(bboxes) - 1, i))],
                    (roi_crf - base_crf) / 51,  # calculate qoffset
                    drawbox=drawbox,
                )
                for i in range(int(duration_s * true_video_fps / update_freq))
            )
        )
        .output(
            out_filename,
            pix_fmt="yuv420p",
            vcodec=CODEC,
            crf=base_crf,
            preset=PRESET,
        )
        .compile(
            overwrite_output=True,
        )
    )

    return subprocess.Popen(args, stdin=subprocess.PIPE)


def read_frame(process1: subprocess.Popen, video_info: VideoInfo):
    width, height = video_info.width, video_info.height
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
    return frame


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
