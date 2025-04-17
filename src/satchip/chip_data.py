import argparse
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
from tqdm import tqdm

import satchip
from satchip import utils
from satchip.chip_sentinel1rtc import get_s1rtc_data
from satchip.chip_sentinel2 import get_s2l2a_data
from satchip.terra_mind_grid import TerraMindGrid


def chip_data(label_path: str, get_data_fn: callable, output_dir: Path) -> Path:
    labels = utils.load_chip(label_path)
    date = labels.time.data[0].astype('M8[ms]').astype(datetime)
    bounds = labels.attrs['bounds']
    grid = TerraMindGrid([bounds[1] - 1, bounds[3] + 1], [bounds[0] - 1, bounds[2] + 1])
    terra_mind_chips = [c for c in grid.terra_mind_chips if c.name in list(labels.sample.data)]

    data_chips = []
    with TemporaryDirectory() as scratch_dir:
        scratch_dir = Path(scratch_dir)
        for chip in tqdm(terra_mind_chips):
            data_chips.append(get_data_fn(chip, date, scratch_dir))

    attrs = {'date_created': date.isoformat(), 'satchip_version': satchip.__version__, 'bounds': labels.attrs['bounds']}
    dataset = xr.Dataset(attrs=attrs)
    # NOTE: may only work when all chips have same date
    dataset['bands'] = xr.concat(data_chips, dim='sample')
    dataset['lats'] = labels['lats']
    dataset['lons'] = labels['lons']
    output_path = output_dir / (label_path.with_suffix('').with_suffix('').name + '_S2.zarr.zip')
    utils.save_chip(dataset, output_path)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('labelpath', type=str, help='Path to the label image')
    parser.add_argument('dataset', choices=['s2l2a', 's1rtc'], type=str, help='Dataset to create chips for')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory for the chips')
    args = parser.parse_args()

    args.labelpath = Path(args.labelpath)
    dataset_fns = {'s2l2a': get_s2l2a_data, 's1rtc': get_s1rtc_data}
    dataset_fn = dataset_fns[args.dataset.lower()]
    args.outdir = Path(args.outdir)

    chip_data(args.labelpath, dataset_fn, args.outdir)


if __name__ == '__main__':
    main()
