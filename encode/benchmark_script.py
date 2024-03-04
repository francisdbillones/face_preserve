import argparse
import subprocess
import pathlib
from tqdm import tqdm

from main import run
from constants import DRAWBOX, NON_ROI_CRF, ROI_CRF


def main():
    parser = argparse.ArgumentParser(description="Batch processing for FacePreserve")

    parser.add_argument("in_dir", help="Input directory")
    parser.add_argument("out_dir", help="Output directory")
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

    in_dir = args.in_dir
    out_dir = args.out_dir
    selective_type = args.selective_type
    base_crf = args.base_crf
    roi_crf = args.roi_crf
    drawbox = args.drawbox

    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)

    for in_filename in tqdm(in_dir.glob("*.mp4")):
        out_filename = out_dir / in_filename.name

        run(
            in_filename.absolute(),
            out_filename.absolute(),
            selective_type,
            base_crf,
            roi_crf,
            drawbox=drawbox,
        )


if __name__ == "__main__":
    main()
