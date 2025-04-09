import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rioxarray
import shapely
import xarray as xr
from pystac_client import Client

from satchip import utils
from satchip.terra_mind_grid import TerraMindChip, TerraMindGrid


def create_template_da(chip: TerraMindChip) -> xr.DataArray:
    x = np.arange(chip.nrow) * chip.gdal_transform[1] + chip.gdal_transform[0] + chip.gdal_transform[1] / 2
    y = np.arange(chip.ncol) * chip.gdal_transform[5] + chip.gdal_transform[3] + chip.gdal_transform[5] / 2
    template = xr.DataArray(np.zeros((chip.ncol, chip.nrow)), dims=('y', 'x'), coords={'y': y, 'x': x})
    template.rio.write_crs(f'EPSG:{chip.epsg}', inplace=True)
    template.rio.write_transform(chip.rio_transform, inplace=True)
    return template


def get_s2l2a_data(chip: TerraMindChip, date: datetime) -> xr.DataArray:
    """Returns XArray DataArray of Sentinel-2 L2A image for the given bounds and
    closest collection after date.

    If multiple images are available, the one with the most coverage is returned.
    """
    date_end = date + timedelta(weeks=1)
    date_range = f'{datetime.strftime(date, "%Y-%m-%d")}/{datetime.strftime(date_end, "%Y-%m-%d")}'
    roi = shapely.box(*chip.bounds)
    client = Client.open('https://earth-search.aws.element84.com/v1')
    search = client.search(
        collections=['sentinel-2-l2a'],
        intersects=roi,
        datetime=date_range,
        max_items=50,
    )
    items = list(search.item_collection())
    items.sort(key=lambda x: x.datetime)
    coverage = []
    for item in search.item_collection():
        image_footprint = shapely.geometry.shape(item.geometry)
        intersection = roi.intersection(image_footprint)
        coverage.append(intersection.area / roi.area)
    item = items[coverage.index(max(coverage))]
    bands = {
        'B01': 'coastal',
        'B02': 'blue',
        'B03': 'green',
        'B04': 'red',
        'B05': 'rededge1',
        'B06': 'rededge2',
        'B07': 'rededge3',
        'B08': 'nir',
        'B8A': 'nir08',
        'B09': 'nir09',
        'B11': 'swir16',
        'B12': 'swir22',
    }
    roi_buffered = roi.buffer(0.1)
    das = []
    template = create_template_da(chip)
    for band in bands:
        href = item.assets[bands[band]].href
        da = rioxarray.open_rasterio(href).rio.clip_box(*roi_buffered.bounds, crs='EPSG:4326')
        da['band'] = [band]
        da_reproj = da.rio.reproject_match(template)
        das.append(da_reproj)
    dataarray = xr.concat(das, dim='band')
    return dataarray


def chip_sentinel2(label_path: str, output_dir: Path) -> Path:
    labels = utils.load_chip(label_path)
    date = labels.time.data[0].astype('M8[ms]').astype(datetime)
    bounds = labels.attrs['bounds']
    grid = TerraMindGrid([bounds[1], bounds[3]], [bounds[0], bounds[2]])
    data_chips = []
    for chip in grid.terra_mind_chips:
        data_chips.append(get_s2l2a_data(chip, date))
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
