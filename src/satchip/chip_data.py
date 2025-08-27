import argparse
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
from tqdm import tqdm

import satchip
from satchip import utils
from satchip.chip_hls import get_hls_data
from satchip.chip_sentinel1rtc import get_s1rtc_data
from satchip.chip_sentinel2 import get_s2l2a_data
from satchip.terra_mind_grid import TerraMindGrid


GET_DATA_FNS = {'S2L2A': get_s2l2a_data, 'S1RTC': get_s1rtc_data, 'HLS': get_hls_data}


def chip_data(label_path: Path, platform: str, output_dir: Path, scratch_dir: Path | None) -> xr.Dataset:
    get_data_fn = GET_DATA_FNS[platform]
    labels = utils.load_chip(label_path)
    date = labels.time.data[0].astype('M8[ms]').astype(datetime)
    bounds = labels.attrs['bounds']
    grid = TerraMindGrid([bounds[1] - 1, bounds[3] + 1], [bounds[0] - 1, bounds[2] + 1])
    terra_mind_chips = [c for c in grid.terra_mind_chips if c.name in list(labels.sample.data)]

    data_chips = []
    if scratch_dir is not None:
        for chip in tqdm(terra_mind_chips):
            data_chips.append(get_data_fn(chip, date, scratch_dir))
    else:
        with TemporaryDirectory() as tmp_dir:
            scratch_dir = Path(tmp_dir)
            for chip in tqdm(terra_mind_chips):
                data_chips.append(get_data_fn(chip, date, scratch_dir))

    attrs = {'date_created': date.isoformat(), 'satchip_version': satchip.__version__, 'bounds': labels.attrs['bounds']}
    dataset = xr.Dataset(attrs=attrs)
    # NOTE: may only work when all chips have same date
    dataset['bands'] = xr.concat(data_chips, dim='sample')
    dataset['lats'] = labels['lats']
    dataset['lons'] = labels['lons']
    output_path = output_dir / (label_path.with_suffix('').with_suffix('').name + f'_{platform}.zarr.zip')
    utils.save_chip(dataset, output_path)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('labelpath', type=Path, help='Path to the label image')
    parser.add_argument('platform', choices=['S2L2A', 'S1RTC', 'HLS'], type=str, help='Dataset to create chips for')
    parser.add_argument('--outdir', default='.', type=Path, help='Output directory for the chips')
    parser.add_argument(
        '--scratchdir', default=None, type=Path, help='Output directory for scratch files if you want to keep them'
    )
    args = parser.parse_args()
    args.platform = args.platform.upper()
    chip_data(args.labelpath, args.platform, args.outdir, args.scratchdir)


if __name__ == '__main__':
    main()
