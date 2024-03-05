from __future__ import print_function
import argparse
from typing import List
import cv2
import ffmpeg
import numpy as np
import subprocess
import itertools

from model_utils import (
    BlazeFaceDetector,
    HOGDetector,
    HaarCascadeDetector,
    iou,
    load_model,
    BBox,
)
from utils import VideoInfo, add_rois, get_video_info
from constants import (
    BLAZEFACE_OPTIONS,
    BLAZEFACE_UPDATE_FREQ,
    HAARCASCADE_UPDATE_FREQ,
    HOG_FULLBODY_OPTIONS,
    HOG_UPDATE_FREQ,
    NON_ROI_CRF,
    ROI_CRF,
    DRAWBOX,
    CODEC,
    PRESET,
)

# helper = lambda obj: print("DEBUG:", obj) or obj
helper = lambda obj: obj


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
                    bboxes[helper(min(len(bboxes) - 1, i))],
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


def run(
    in_filename: str,
    out_filename: str,
    selective_type: str,
    roi_crf: int,
    decayed_crf: int,
    drawbox: bool,
):
    assert roi_crf < decayed_crf
    assert 0 <= decayed_crf <= 51
    assert 0 <= roi_crf <= 51

    assert selective_type in ["faceshortrange"]  # , "facelongrange", "fullbody"]
    if selective_type == "faceshortrange":
        model = BlazeFaceDetector(BLAZEFACE_OPTIONS)

    # elif selective_type == "facelongrange":
    #     model = HaarCascadeDetector(
    #         HAARCASCADE_FACE_PATH,
    #         options=HAARCASCADE_FACE_OPTIONS,
    #     )

    # elif selective_type == "fullbody":
    #     hog_descriptor = cv2.HOGDescriptor()
    #     hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #     model = HOGDetector(
    #         hog_descriptor,
    #         HOG_FULLBODY_OPTIONS,
    #     )

    update_freq = {
        "faceshortrange": BLAZEFACE_UPDATE_FREQ,
        "facelongrange": HAARCASCADE_UPDATE_FREQ,
        "fullbody": HOG_UPDATE_FREQ,
    }[selective_type]

    video_info = get_video_info(in_filename)
    video_to_rgb24_process = ffmpeg_video_to_rgb24_process(in_filename)

    bboxes = []
    for frame_count in itertools.count(start=0):
        in_frame = read_frame(video_to_rgb24_process, video_info)
        if in_frame is None:
            break

        if frame_count % update_freq == 0:
            # when blazeface is running in video mode, it needs an additional parameter
            # of the frame timestamp in milliseconds
            if selective_type == "faceshortrange":
                frame_timestamp_ms = int(frame_count * 1000 / video_info.fps)
                bboxes.append(model.detect(in_frame, frame_timestamp_ms))
            else:
                bboxes.append(model.detect(in_frame))

    add_bboxes_process = ffmpeg_add_bboxes_process(
        in_filename,
        out_filename,
        decayed_crf,
        roi_crf,
        bboxes,
        update_freq,
        video_info,
        drawbox,
    )

    video_to_rgb24_process.wait()

    add_bboxes_process.stdin.close()
    add_bboxes_process.wait()


def main():
    parser = argparse.ArgumentParser(
        description="FacePreserve: Selective Compression for Video Calls"
    )
    parser.add_argument("in_filename", help="Input filename")
    parser.add_argument("out_filename", help="Output filename")
    # add argument --selective_type, where selective_type is either "faceshortrange", "facelongrange", or "fullbody"
    parser.add_argument(
        "--selective_type",
        help="Type of selective compression",
        choices=["faceshortrange"],  # , "facelongrange", "fullbody"],
        default="faceshortrange",
    )
    parser.add_argument(
        "--base_crf",
        help="CRF for non-ROI",
        type=int,
        default=NON_ROI_CRF,
    )
    parser.add_argument(
        "--roi_crf",
        help="CRF for ROI",
        type=int,
        default=ROI_CRF,
    )
    parser.add_argument(
        "--drawbox",
        help="Draw bounding boxes",
        action="store_true",
        default=DRAWBOX,
    )

    args = parser.parse_args()
    in_filename = args.in_filename
    out_filename = args.out_filename
    selective_type = args.selective_type
    base_crf = args.base_crf
    roi_crf = args.roi_crf
    drawbox = args.drawbox
    run(in_filename, out_filename, selective_type, roi_crf, base_crf, drawbox)


if __name__ == "__main__":
    main()
