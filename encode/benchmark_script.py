import argparse
import subprocess
import pathlib
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Batch processing for FacePreserve")

parser.add_argument("in_dir", help="Input directory")
parser.add_argument("out_dir", help="Output directory")
parser.add_argument(
    "--selective_type",
    help="Type of selective compression",
    choices=["faceshortrange", "facelongrange", "fullbody"],
    default="faceshortrange",
)
parser.add_argument(
    "--base_crf",
    help="CRF for non-ROI",
    type=int,
    default=30,
)
parser.add_argument(
    "--roi_crf",
    help="CRF for ROI",
    type=int,
    default=35,
)


def main():
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    selective_type = args.selective_type
    base_crf = args.base_crf
    roi_crf = args.roi_crf

    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)

    for in_filename in tqdm(in_dir.glob("*.mp4")):
        out_filename = out_dir / in_filename.name
        subprocess.run(
            [
                "python3",
                "main.py",
                in_filename.absolute(),
                out_filename.absolute(),
                "--selective_type",
                selective_type,
                "--base_crf",
                str(base_crf),
                "--roi_crf",
                str(roi_crf),
            ]
        )


if __name__ == "__main__":
    main()
