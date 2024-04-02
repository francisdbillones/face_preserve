from __future__ import print_function
import argparse
from typing import List
import itertools

from model_utils import (
    GET_MODEL,
)
from utils import (
    get_video_info,
    read_frame,
    ffmpeg_video_to_rgb24_process,
    ffmpeg_add_bboxes_process,
)
from constants import (
    UPDATE_FREQ,
    NON_ROI_CRF,
    ROI_CRF,
    DRAWBOX,
)


def run(
    in_filename: str,
    out_filename: str,
    roi_crf: int,
    decayed_crf: int,
    update_freq: int,
    drawbox: bool,
):
    assert roi_crf < decayed_crf
    assert 0 <= decayed_crf <= 51
    assert 0 <= roi_crf <= 51

    # modify this to use the model you want
    model = GET_MODEL()

    video_info = get_video_info(in_filename)
    video_to_rgb24_process = ffmpeg_video_to_rgb24_process(in_filename)

    bboxes = []
    for frame_count in itertools.count(start=0):
        in_frame = read_frame(video_to_rgb24_process, video_info)
        if in_frame is None:
            break

        if frame_count % update_freq == 0:
            # modify below to use the model inference scheme you need

            # when blazeface is running in video mode, it needs an additional parameter
            # of the frame timestamp in milliseconds
            frame_timestamp_ms = int(frame_count * 1000 / video_info.fps)
            bboxes.append(model.detect(in_frame, frame_timestamp_ms))

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
    parser.add_argument("in_filename", help="Input filename", type=str)
    parser.add_argument("out_filename", help="Output filename", type=str)
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
        "--update_freq",
        help="Number of frames to wait before updating bounding boxes",
        type=int,
        default=UPDATE_FREQ,
    )
    parser.add_argument(
        "--drawbox",
        help="Draw bounding boxes",
        action="store_true",
        default=DRAWBOX,
    )

    args = parser.parse_args()
    in_filename: str = args.in_filename
    out_filename: str = args.out_filename
    base_crf: int = args.base_crf
    roi_crf: int = args.roi_crf
    update_freq: int = args.update_freq
    drawbox: bool = args.drawbox
    run(in_filename, out_filename, roi_crf, base_crf, update_freq, drawbox)


if __name__ == "__main__":
    main()
