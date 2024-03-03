import pandas as pd
import ffmpeg
from tqdm import tqdm
from pathlib import Path

dataframes = {}
directory = Path("./benchmarking/video_call_mos_set/")

for subdir in tqdm(sorted(directory.iterdir())):
    data = pd.DataFrame(
        columns=["video_name", "duration", "resolution", "fps", "bitrate", "size"]
    )

    for file in tqdm(sorted(subdir.iterdir())):
        probe = ffmpeg.probe(str(file.absolute()))
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        video_name = file.name
        duration = float(video_stream["duration"])
        resolution = f"{video_stream['width']}x{video_stream['height']}"
        fps = eval(video_stream["r_frame_rate"])
        bitrate = video_stream["bit_rate"]
        size = file.stat().st_size

        # add row to the dataframe
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    [[video_name, duration, resolution, fps, bitrate, size]],
                    columns=[
                        "video_name",
                        "duration",
                        "resolution",
                        "fps",
                        "bitrate",
                        "size",
                    ],
                ),
            ]
        )

    dataframes[subdir.name] = data


with pd.ExcelWriter("data.xlsx") as writer:
    for key in dataframes:
        dataframes[key].to_excel(writer, sheet_name=key, index=False)
