from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rioxarray
import s3fs
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

S3_FS = s3fs.S3FileSystem(anon=True)


def url_to_s3path(url: str) -> str:
    parsed = urlparse(url)
    netloc_parts = parsed.netloc.split('.')
    if 's3' in netloc_parts:
        bucket = netloc_parts[0]
    else:
        raise ValueError(f'URL in not an S3 URL: {url}')
    key = parsed.path.lstrip('/')
    return f'{bucket}/{key}'


def url_to_localpath(url: str, scratch_dir: Path) -> Path:
    parsed = urlparse(url)
    name = '_'.join(parsed.path.lstrip('/').split('/')[-2:])
    local_file_path = scratch_dir / name
    return local_file_path


def fetch_s3_file(url: str, scratch_dir: Path) -> Path:
    local_path = url_to_localpath(url, scratch_dir)
    if not local_path.exists():
        s3_path = url_to_s3path(url)
        S3_FS.get(s3_path, str(local_path))
    return local_path


def multithread_fetch_s3_file(urls: list[str], scratch_dir: Path, max_workers: int = 8) -> None:
    s3_paths, download_paths = [], []
    for url in urls:
        local_path = url_to_localpath(url, scratch_dir)
        if not local_path.exists():
            download_paths.append(local_path)
            s3_paths.append(url_to_s3path(url))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(S3_FS.get, s3_paths, download_paths)


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
    multithread_fetch_s3_file([item.assets[S2_BANDS[band]].href for band in S2_BANDS], scratch_dir)
    template = create_template_da(chip)
    das = []
    for band in S2_BANDS:
        local_path = url_to_localpath(item.assets[S2_BANDS[band]].href, scratch_dir)
        assert local_path.exists(), f'File not found: {local_path}'
        da = rioxarray.open_rasterio(local_path).rio.clip_box(*roi_buffered.bounds, crs='EPSG:4326')
        da['band'] = [band]
        da_reproj = da.rio.reproject_match(template)
        das.append(da_reproj)
    dataarray = xr.concat(das, dim='band').drop_vars('spatial_ref')
    dataarray['x'] = np.arange(0, chip.ncol)
    dataarray['y'] = np.arange(0, chip.nrow)
    dataarray = dataarray.expand_dims({'time': [item.datetime.replace(tzinfo=None)], 'sample': [chip.name]})
    dataarray.attrs = {}
    return dataarray
