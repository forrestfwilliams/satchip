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
    'B01': 'COASTAL',
    'B02': 'BLUE',
    'B03': 'GREEN',
    'B04': 'RED',
    'B05': 'REDEDGE1',
    'B06': 'REDEDGE2',
    'B07': 'REDEDGE3',
    'B08': 'NIR',
    'B8A': 'NIR08',
    'B09': 'NIR09',
    'B11': 'SWIR16',
    'B12': 'SWIR22',
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


def get_scenes(
    items: list[Item], roi: shapely.geometry.Polygon, strategy: str, max_cloud_pct: int, scratch_dir: Path
) -> list[Item]:
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
    strategy = strategy.upper()
    assert strategy in ['BEST', 'ALL'], 'Strategy must be either BEST or ALL'
    assert len(items) > 0, 'No Sentinel-2 L2A scenes found for chip.'
    items = [item for item in items if get_pct_intersect(item.geometry, roi) > 0.95]
    best_first = sorted(items, key=lambda x: (-get_pct_intersect(x.geometry, roi), x.datetime))
    valid_scenes = []
    for item in best_first:
        scl_href = item.assets['scl'].href
        local_path = fetch_s3_file(scl_href, scratch_dir)
        assert local_path.exists(), f'File not found: {local_path}'
        scl_da = rioxarray.open_rasterio(local_path).rio.clip_box(*roi.bounds, crs='EPSG:4326')  # type: ignore
        scl_array = scl_da.data[0]
        # Looks for nodata (0), defective pixels (1), cloud shadows (3), clouds (8/9), cirrus (10)
        # See https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
        # for details on SCL values
        bad_pixels = np.isin(scl_array, [0, 1, 3, 8, 9, 10])
        pct_bad = int(np.round(np.sum(bad_pixels) / bad_pixels.size * 100))
        if pct_bad <= max_cloud_pct:
            if strategy == 'BEST':
                return [item]
            else:
                valid_scenes.append(item)

    assert len(valid_scenes) > 0, f'No Sentinel-2 L2A scenes found with <={max_cloud_pct}% cloud cover for chip.'
    return valid_scenes


def get_s2l2a_data(chip: TerraMindChip, scratch_dir: Path, opts: dict) -> xr.DataArray:
    """Get XArray DataArray of Sentinel-2 L2A image for the given bounds and best collection parameters.

    Args:
        chip: TerraMindChip object defining the area of interest.
        scratch_dir: Directory to store downloaded files.
        opts: Options dictionary with the following keys
            - date_start: Start date for the search.
            - date_end: End date for the search.
            - strategy (optional): Strategy to use when multiple scenes are found.
            - max_cloud_pct (optional): Maximum percent of bad pixels allowed in the scene.

    Returns:
        XArray DataArray containing the Sentinel-2 L2A image data.
    """
    date_start = opts['date_start']
    date_end = opts['date_end'] + timedelta(days=1)  # inclusive end
    date_range = f'{datetime.strftime(date_start, "%Y-%m-%d")}/{datetime.strftime(date_end, "%Y-%m-%d")}'
    roi = shapely.box(*chip.bounds)
    roi_buffered = roi.buffer(0.01)
    client = Client.open('https://earth-search.aws.element84.com/v1')
    search = client.search(
        collections=['sentinel-2-l2a'],
        intersects=roi,
        datetime=date_range,
        max_items=1000,
    )
    assert len(search.item_collection()) > 0, (
        f'No Sentinel-2 L2A scenes found for chip {chip.name} between {date_start} and {date_end}.'
    )
    items = list(search.item_collection())
    max_cloud_pct = opts.get('max_cloud_pct', 100)
    strategy = opts.get('strategy', 'BEST')
    items = get_scenes(items, roi, strategy, max_cloud_pct, scratch_dir)
    urls = [item.assets[S2_BANDS[band].lower()].href for item in items for band in S2_BANDS]
    multithread_fetch_s3_file(urls, scratch_dir)
    template = create_template_da(chip)
    das = []
    for item in items:
        for band in S2_BANDS:
            local_path = url_to_localpath(item.assets[S2_BANDS[band].lower()].href, scratch_dir)
            assert local_path.exists(), f'File not found: {local_path}'
            da = rioxarray.open_rasterio(local_path).rio.clip_box(*roi_buffered.bounds, crs='EPSG:4326')  # type: ignore
            da_reproj = da.rio.reproject_match(template)
            da_reproj['band'] = [S2_BANDS[band]]
            da_reproj = da_reproj.expand_dims({'time': [item.datetime.replace(tzinfo=None)]})  # type: ignore
            da_reproj['x'] = np.arange(0, chip.ncol)
            da_reproj['y'] = np.arange(0, chip.nrow)
            da_reproj.attrs = {}
            das.append(da_reproj)
    dataarray = xr.combine_by_coords(das, join='override').drop_vars('spatial_ref')
    assert isinstance(dataarray, xr.DataArray)
    dataarray = dataarray.expand_dims({'sample': [chip.name], 'platform': ['S2L2A']})
    dataarray.attrs = {}
    return dataarray
