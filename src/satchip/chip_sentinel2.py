from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rioxarray
import shapely
import xarray as xr
from pystac_client import Client

from satchip.chip_xr_base import create_template_da
from satchip.terra_mind_grid import TerraMindChip


S2_BANDS = {
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


def get_s2l2a_data(chip: TerraMindChip, date: datetime, scratch_dir: Path) -> xr.DataArray:
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
    roi_buffered = roi.buffer(0.1)
    das = []
    template = create_template_da(chip)
    for band in S2_BANDS:
        href = item.assets[S2_BANDS[band]].href
        da = rioxarray.open_rasterio(href).rio.clip_box(*roi_buffered.bounds, crs='EPSG:4326')
        da['band'] = [band]
        da_reproj = da.rio.reproject_match(template)
        das.append(da_reproj)
    dataarray = xr.concat(das, dim='band').drop_vars('spatial_ref')
    dataarray['x'] = np.arange(0, chip.ncol)
    dataarray['y'] = np.arange(0, chip.nrow)
    dataarray = dataarray.expand_dims({'time': [item.datetime.replace(tzinfo=None)], 'sample': [chip.name]})
    dataarray.attrs = {}
    return dataarray
