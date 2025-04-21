import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio as rio
import xarray as xr
from tqdm import tqdm

import satchip
from satchip import utils
from satchip.terra_mind_grid import TerraMindGrid


def get_overall_bounds(bounds: list) -> list:
    minx = min([b[0] for b in bounds])
    miny = min([b[1] for b in bounds])
    maxx = max([b[2] for b in bounds])
    maxy = max([b[3] for b in bounds])
    return [minx, miny, maxx, maxy]


def is_valuable(chip: np.ndarray) -> bool:
    vals = list(np.unique(chip))
    return not vals == [0]


def chip_labels(label_path: Path, date: datetime, output_dir: Path) -> Path:
    label = xr.open_dataarray(label_path)
    bbox = utils.get_epsg4326_bbox(label.rio.bounds(), label.rio.crs.to_epsg())
    tm_grid = TerraMindGrid(latitude_range=(bbox[1], bbox[3]), longitude_range=(bbox[0], bbox[2]))
    chips = {}
    for tm_chip in tqdm(tm_grid.terra_mind_chips):
        chip = label.rio.reproject(
            dst_crs=f'EPSG:{tm_chip.epsg}',
            resampling=rio.enums.Resampling(1),
            transform=tm_chip.rio_transform,
            shape=(tm_chip.nrow, tm_chip.ncol),
        )
        chip_array = chip.data[0]
        chip_array[np.isnan(chip_array)] = 0
        chip_array = np.round(chip_array).astype(np.int16)
        if is_valuable(chip_array):
            chips[tm_chip.name] = [chip_array, tm_chip]

    if len(chips) == 0:
        raise ValueError(f'No valid chips found for {label_path.name}')

    coords = {
        'time': np.array([date]),
        'band': np.array(['labels']),
        'sample': np.array([str(x) for x in chips.keys()]),
        'y': np.arange(0, chip_array.shape[0]),
        'x': np.arange(0, chip_array.shape[1]),
    }
    print(f'Found {len(chips)} valid chips for {label_path.name}')
    label_np = np.expand_dims(np.stack([val[0] for val in chips.values()], axis=0), axis=[0, 1])
    lats, lons = zip(*[val[1].center for val in chips.values()])

    dataset = xr.Dataset(attrs={'date_created': date.isoformat(), 'satchip_version': satchip.__version__})
    dataset.attrs['bounds'] = get_overall_bounds([val[1].bounds for val in chips.values()])
    dataset['bands'] = xr.DataArray(label_np, coords=coords, dims=list(coords.keys()))
    dataset['lats'] = xr.DataArray(np.array(lats), coords={'sample': coords['sample']}, dims=['sample'])
    dataset['lons'] = xr.DataArray(np.array(lons), coords={'sample': coords['sample']}, dims=['sample'])
    output_path = output_dir / label_path.with_suffix('.zarr.zip').name
    utils.save_chip(dataset, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('labelpath', type=str, help='Path to the label image')
    parser.add_argument('date', type=str, help='Date and time of the image in ISO format (YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory for the chips')
    args = parser.parse_args()
    args.labelpath = Path(args.labelpath)
    args.date = datetime.fromisoformat(args.date)
    args.outdir = Path(args.outdir)
    chip_labels(args.labelpath, args.date, args.outdir)


if __name__ == '__main__':
    main()
