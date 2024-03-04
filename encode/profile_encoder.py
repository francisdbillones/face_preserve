# TODO: record the time overhead of BlazeFace as well as the difference in encoding times with just base CRF and just
# decayed crf vs FacePreserve's decayed CRF and base CRF for ROI regions.

import time
import tempfile
import subprocess

from pathlib import Path
from tqdm import tqdm

import pandas as pd
import ffmpeg

from main import run, ffmpeg_video_to_rgb24_process
from constants import DRAWBOX, CODEC, PRESET

CRFS = [20, 25, 30]
INCREMENTS = [5, 10, 15]


def main():
    facepreserve_times = []
    raw_times = []

    raw_video_directory = Path("./benchmarking/video_call_mos_set/raw/")

    for video in tqdm(sorted(raw_video_directory.iterdir())):
        for base_crf in CRFS:
            for increment in INCREMENTS:
                decayed_crf = base_crf + increment
                # record times for encoding with just decayed CRF
                with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_output:
                    start_time = time.perf_counter()
                    args = (
                        ffmpeg.input(video.absolute())
                        .output(
                            tmp_output.name,
                            pix_fmt="yuv420p",
                            vcodec=CODEC,
                            crf=base_crf + increment,
                            preset=PRESET,
                        )
                        .compile(overwrite_output=True)
                    )

                    subprocess.Popen(args).wait()

                    end_time = time.perf_counter()
                raw_encoding_time = end_time - start_time

                # record times for encoding raw video with base_crf
                with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_output:
                    start_time = time.perf_counter()
                    run(
                        video.absolute(),
                        tmp_output.name,
                        "faceshortrange",
                        base_crf,
                        decayed_crf,
                        drawbox=DRAWBOX,
                    )
                    end_time = time.perf_counter()

                total_encoding_time = end_time - start_time

                # record time it takes for video_to_rgb24_process to complete
                start_time = time.perf_counter()
                with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_output:
                    ffmpeg_video_to_rgb24_process(
                        video.absolute(), pipe=subprocess.DEVNULL
                    ).wait()
                end_time = time.perf_counter()
                video_to_rgb24_time = end_time - start_time

                corrected_encoding_time = total_encoding_time - video_to_rgb24_time

                facepreserve_times.append(
                    {
                        "video": video.name,
                        "base_crf": base_crf,
                        "decayed_crf": decayed_crf,
                        "elapsed_time": corrected_encoding_time,
                    }
                )
                raw_times.append(
                    {
                        "video": video.name,
                        "crf": base_crf + increment,
                        "elapsed_time": raw_encoding_time,
                    }
                )

    facepreserve_times_df = pd.DataFrame(facepreserve_times)
    facepreserve_times_df.to_xlsx("facepreserve_encoding_times.xlsx", index=False)

    raw_times_df = pd.DataFrame(raw_times)
    raw_times_df.to_xlsx("raw_encoding_times.xlsx", index=False)


if __name__ == "__main__":
    main()
