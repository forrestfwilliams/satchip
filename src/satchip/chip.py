import argparse
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio as rio
import xarray as xr
from pyproj import CRS, Transformer
from tqdm import tqdm

import satchip
from satchip.terra_mind_grid import TerraMindGrid


def get_epsg4326_bbox(bounds: list, in_epsg: int, buffer: float = 0.1) -> tuple:
    if in_epsg == 4326:
        return bounds
    in_crs = CRS.from_epsg(in_epsg)
    out_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)
    minx, miny = transformer.transform(bounds[0], bounds[1])
    maxx, maxy = transformer.transform(bounds[2], bounds[3])
    minx, miny, maxx, maxy = minx - buffer, miny - buffer, maxx + buffer, maxy + buffer
    return minx, miny, maxx, maxy


def is_valuable(chip: np.ndarray) -> bool:
    vals = list(np.unique(chip))
    return not vals == [0]


def chip_labels(label_path: Path, date: datetime, output_dir: Path) -> Path:
    label = xr.open_dataarray(label_path)
    bbox = get_epsg4326_bbox(label.rio.bounds(), label.rio.crs.to_epsg())
    tm_grid = TerraMindGrid(latitude_range=(bbox[1], bbox[3]), longitude_range=(bbox[0], bbox[2]))
    chips = {}
    for tm_chip in tqdm(tm_grid.terra_mind_chips[1500:1750]):
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
            chips[tm_chip.name] = chip_array
    coords = {
        'time': [date],
        'band': ['labels'],
        'sample': list(chips.keys()),
        'y': np.arange(0, chip_array.shape[0]),
        'x': np.arange(0, chip_array.shape[1]),
    }
    np_array = np.expand_dims(np.stack(list(chips.values()), axis=0), axis=[0, 1])
    data_array = xr.DataArray(np_array, coords=coords, dims=coords.keys())
    dataset = xr.Dataset(attrs={'data_created': date.isoformat(), 'satchip_version': satchip.__version__})
    dataset['bands'] = data_array
    output_path = output_dir / label_path.with_suffix('.zarr').name
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / 'tmp.zarr'
        dataset.to_zarr(tmp_path)
        shutil.make_archive(output_path, 'zip', tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip an image')
    parser.add_argument('label_path', type=str, help='Path to the label image')
    parser.add_argument('date', type=str, help='Date and time of the image in ISO format (YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--output_dir', default='.', type=str, help='Output directory for the chips')
    args = parser.parse_args()
    args.label_path = Path(args.label_path)
    args.date = datetime.fromisoformat(args.date)
    args.output_dir = Path(args.output_dir)
    chip_labels(args.label_path, args.date, args.output_dir)


if __name__ == '__main__':
    main()
