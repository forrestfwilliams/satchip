from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rioxarray
import s3fs
import shapely
import xarray as xr
from pystac.item import Item
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
    """Converts an S3 URL to an S3 path usable by s3fs."""
    parsed = urlparse(url)
    netloc_parts = parsed.netloc.split('.')
    if 's3' in netloc_parts:
        bucket = netloc_parts[0]
    else:
        raise ValueError(f'URL in not an S3 URL: {url}')
    key = parsed.path.lstrip('/')
    return f'{bucket}/{key}'


def url_to_localpath(url: str, scratch_dir: Path) -> Path:
    """Converts an S3 URL to a local file path in the given scratch directory."""
    parsed = urlparse(url)
    name = '_'.join(parsed.path.lstrip('/').split('/')[-2:])
    local_file_path = scratch_dir / name
    return local_file_path


def fetch_s3_file(url: str, scratch_dir: Path) -> Path:
    """Fetches an S3 file to the given scratch directory if it doesn't already exist."""
    local_path = url_to_localpath(url, scratch_dir)
    if not local_path.exists():
        s3_path = url_to_s3path(url)
        S3_FS.get(s3_path, str(local_path))
    return local_path


def multithread_fetch_s3_file(urls: list[str], scratch_dir: Path, max_workers: int = 8) -> None:
    """Fetches multiple S3 files to the given scratch directory using multithreading."""
    s3_paths, download_paths = [], []
    for url in urls:
        local_path = url_to_localpath(url, scratch_dir)
        if not local_path.exists():
            download_paths.append(local_path)
            s3_paths.append(url_to_s3path(url))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(S3_FS.get, s3_paths, download_paths)


def get_pct_intersect(scene_geom: dict | None, roi: shapely.geometry.Polygon) -> float:
    """Returns the percent of the roi polygon that intersects with the scene geometry."""
    if scene_geom is None:
        return 0.0
    image_footprint = shapely.geometry.shape(scene_geom)
    intersection = roi.intersection(image_footprint)
    return intersection.area / roi.area


def get_best_scene(items: list[Item], roi: shapely.geometry.Polygon, max_cloud_pct: int, scratch_dir: Path) -> Item:
    """Returns the best Sentinel-2 L2A scene from the given list of items.
    The best scene is defined as the earliest scene with the largest intersection with the roi and
    less than or equal to the max_cloud_pct of bad pixels (nodata, defective, cloud).

    Args:
        items: List of Sentinel-2 L2A items.
        roi: Region of interest polygon.
        max_cloud_pct: Maximum percent of bad pixels allowed in the scene.
        scratch_dir: Directory to store downloaded files.

    Returns:
        The best Sentinel-2 L2A item.
    """
    assert len(items) > 0, 'No Sentinel-2 L2A scenes found for chip.'
    best_first = sorted(
        items,
        key=lambda x: (
            -get_pct_intersect(x.geometry, roi),  # negative for largest intersect first
            x.datetime,  # earliest date first
        ),
    )
    for item in best_first:
        scl_href = item.assets['scl'].href
        local_path = url_to_localpath(scl_href, scratch_dir)
        assert local_path.exists(), f'File not found: {local_path}'
        scl_da = rioxarray.open_rasterio(local_path).rio.clip_box(*roi.bounds, crs='EPSG:4326')  # type: ignore
        scl_array = scl_da.data[0]
        # Looks for nodata (0), defective pixels (1), cloud medium/high probability (8/9), thin cirrsu (10)
        # See https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
        # for details on SCL values
        bad_pixels = np.isin(scl_array, [0, 1, 8, 9, 10])
        pct_bad = int(np.round(np.sum(bad_pixels) / bad_pixels.size * 100))
        if pct_bad <= max_cloud_pct:
            return item
    raise ValueError(f'No Sentinel-2 L2A scenes found with <= {max_cloud_pct}% cloud cover for chip.')


def get_s2l2a_data(chip: TerraMindChip, date: datetime, scratch_dir: Path, opts:dict) -> xr.DataArray:
    """Get XArray DataArray of Sentinel-2 L2A image for the given bounds and best collection parameters.

    Args:
        chip: TerraMindChip object defining the area of interest.
        date: Date to search for the closest Sentinel-2 L2A image.
        scratch_dir: Directory to store downloaded files.
        opts: Additional options

    Returns:
        XArray DataArray containing the Sentinel-2 L2A image data.
    """
    date_end = date + timedelta(weeks=1)
    date_range = f'{datetime.strftime(date, "%Y-%m-%d")}/{datetime.strftime(date_end, "%Y-%m-%d")}'
    roi = shapely.box(*chip.bounds)
    roi_buffered = roi.buffer(0.01)
    client = Client.open('https://earth-search.aws.element84.com/v1')
    search = client.search(
        collections=['sentinel-2-l2a'],
        intersects=roi,
        datetime=date_range,
        max_items=50,
    )
    items = list(search.item_collection())
    max_cloud_pct = opts.get('max_cloud_pct', 100)
    item = get_best_scene(items, roi, max_cloud_pct, scratch_dir)
    multithread_fetch_s3_file([item.assets[S2_BANDS[band]].href for band in S2_BANDS], scratch_dir)
    template = create_template_da(chip)
    das = []
    for band in S2_BANDS:
        local_path = url_to_localpath(item.assets[S2_BANDS[band]].href, scratch_dir)
        assert local_path.exists(), f'File not found: {local_path}'
        da = rioxarray.open_rasterio(local_path).rio.clip_box(*roi_buffered.bounds, crs='EPSG:4326')  # type: ignore
        da['band'] = [band]
        da_reproj = da.rio.reproject_match(template)
        das.append(da_reproj)
    dataarray = xr.concat(das, dim='band').drop_vars('spatial_ref')
    dataarray['x'] = np.arange(0, chip.ncol)
    dataarray['y'] = np.arange(0, chip.nrow)
    dataarray = dataarray.expand_dims({'time': [item.datetime.replace(tzinfo=None)], 'sample': [chip.name]})  # type: ignore
    dataarray.attrs = {}
    return dataarray
