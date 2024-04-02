import itertools
from pathlib import Path

import pandas as pd
import ffmpeg
import numpy as np
from tqdm import tqdm

from utils import get_video_info
from main import ffmpeg_video_to_rgb24_process, read_frame
from constants import BLAZEFACE_OPTIONS, BLAZEFACE_UPDATE_FREQ
from model_utils import BBox, BlazeFaceDetector


def main():
    dataframe = pd.DataFrame(columns=["video_name", "roi_percentage"])
    directory = Path("./benchmarking/video_call_mos_set/raw")

    for file in tqdm(sorted(directory.iterdir())):
        video_name = file.name

        roi_percentage = calculate_roi_percentage(file)

        # add row to the dataframe
        dataframe = pd.concat(
            [
                dataframe,
                pd.DataFrame(
                    [[video_name, roi_percentage]],
                    columns=["video_name", "roi_percentage"],
                ),
            ]
        )

    with pd.ExcelWriter("roi_percentage_data.xlsx") as writer:
        dataframe.to_excel(writer)


def calculate_roi_percentage(file: Path) -> float:
    model = BlazeFaceDetector(BLAZEFACE_OPTIONS)
    video_info = get_video_info(str(file.absolute()))
    video_to_rgb24_process = ffmpeg_video_to_rgb24_process(str(file.absolute()))

    bboxes = []

    for frame_count in itertools.count(start=0):
        in_frame = read_frame(video_to_rgb24_process, video_info)
        if in_frame is None:
            break

        if frame_count % BLAZEFACE_UPDATE_FREQ == 0:
            frame_timestamp_ms = int(frame_count * 1000 / video_info.fps)
            bboxes.append(model.detect(in_frame, frame_timestamp_ms))

    buffer = np.zeros((video_info.height, video_info.width), dtype=np.uint8)
    roi_percentages = np.zeros(len(bboxes), dtype=np.float32)
    for i, frame_bboxes in enumerate(bboxes):
        if len(frame_bboxes) == 1:
            bbox = frame_bboxes[0]
            frame_roi_percentage = (
                bbox.w * bbox.h / (video_info.height * video_info.width)
            )
        else:
            for bbox in frame_bboxes:
                bbox: BBox
                buffer[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w] = 1

            frame_roi_percentage = np.sum(buffer) / (
                video_info.height * video_info.width
            )
            buffer.fill(0)
        roi_percentages[i] = frame_roi_percentage

    return roi_percentages.mean()


if __name__ == "__main__":
    main()
