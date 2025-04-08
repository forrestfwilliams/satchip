import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from satchip import utils
from satchip.major_tom_grid import MajorTomGrid


def get_data(sample: str, date: datetime) -> np.ndarray:
    return None


def chip_sentinel2(label_path: str, output_dir: Path) -> Path:
    labels = utils.load_chip(label_path)
    major_tom_chips = list(set([x[0:9] for x in labels.coords['sample'].data]))
    date = labels.time.data[0].astype('M8[ms]').astype(datetime)
    data = get_data(major_tom_chips[0], date)
    # get sample name
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('labelpath', type=str, help='Path to the label image')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory for the chips')
    args = parser.parse_args()
    args.labelpath = Path(args.labelpath)
    args.outdir = Path(args.outdir)
    chip_sentinel2(args.labelpath, args.outdir)


if __name__ == '__main__':
    main()
