from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import numpy as np
import rioxarray
import shapely
import xarray as xr

from satchip.chip_xr_base import create_template_da
from satchip.terra_mind_grid import TerraMindChip


HLS_L_BANDS = {
    'B01': 'coastal',
    'B02': 'blue',
    'B03': 'green',
    'B04': 'red',
    'B05': 'nir08',
    'B06': 'swir16',
    'B07': 'swir22',
}
HLS_S_BANDS = {
    'B01': 'coastal',
    'B02': 'blue',
    'B03': 'green',
    'B04': 'red',
    'B8A': 'nir08',
    'B11': 'swir16',
    'B12': 'swir22',
}
BAND_SETS = {'L30': HLS_L_BANDS, 'S30': HLS_S_BANDS}


def get_pct_cloud(umm: dict) -> float:
    return [float(x['Values'][0]) for x in umm['AdditionalAttributes'] if x['Name'] == 'CLOUD_COVERAGE'][0]


def get_pct_intersect(umm: dict, roi: shapely.geometry.Polygon) -> float:
    points = umm['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]['Boundary']['Points']
    coords = [(pt['Longitude'], pt['Latitude']) for pt in points]
    image_roi = shapely.geometry.Polygon(coords)
    return roi.intersection(image_roi).area / roi.area


def get_date(umm: dict) -> datetime:
    date_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
    date = [x['Values'][0] for x in umm['AdditionalAttributes'] if x['Name'] == 'SENSING_TIME'][0]
    return datetime.strptime(date, date_fmt)


def get_product_id(umm: dict) -> str:
    return [x['Identifier'] for x in umm['DataGranule']['Identifiers'] if x['IdentifierType'] == 'ProducerGranuleId'][0]


def get_hls_data(chip: TerraMindChip, date: datetime, scratch_dir: Path, opts: dict) -> xr.DataArray:
    """Returns XArray DataArray of a Harmonized Landsat Sentinel-2 image for the given bounds and
    closest collection after date.

    If multiple images are available, the one with the most coverage is returned.
    """
    date_end = date + timedelta(weeks=1)
    date_start = f'{datetime.strftime(date, "%Y-%m-%d")}'
    date_end = f'{datetime.strftime(date_end, "%Y-%m-%d")}'
    earthaccess.login()
    results = earthaccess.search_data(
        short_name=['HLSL30', 'HLSS30'], bounding_box=chip.bounds, temporal=(date_start, date_end)
    )
    few_clouds = [x for x in results if get_pct_cloud(x['umm']) <= 25]
    if len(few_clouds) == 0:
        raise ValueError('No HLS scenes found with <= 25% cloud cover for dataset.')
    roi = shapely.box(*chip.bounds)
    roi_buffered = roi.buffer(0.1)
    few_clouds = sorted(
        few_clouds,
        key=lambda x: (
            -get_pct_intersect(x['umm'], roi),  # negative for largest intersect first
            get_date(x['umm']),  # earliest date first
        ),
    )
    best_scene = few_clouds[0]
    product_id = get_product_id(best_scene['umm'])
    n_products = len(list(scratch_dir.glob(f'{product_id}*')))
    if n_products < 18:
        earthaccess.download([best_scene], scratch_dir)
    das = []
    template = create_template_da(chip)
    bands = BAND_SETS[product_id.split('.')[1]]
    for band in bands:
        image_path = scratch_dir / f'{product_id}.v2.0.{band}.tif'
        da = rioxarray.open_rasterio(image_path).rio.clip_box(*roi_buffered.bounds, crs='EPSG:4326')  # type: ignore
        da['band'] = [bands[band]]
        da_reproj = da.rio.reproject_match(template)
        das.append(da_reproj)
    dataarray = xr.concat(das, dim='band').drop_vars('spatial_ref')
    dataarray['x'] = np.arange(0, chip.ncol)
    dataarray['y'] = np.arange(0, chip.nrow)
    dataarray = dataarray.expand_dims(
        {'time': [get_date(best_scene['umm']).replace(tzinfo=None)], 'sample': [chip.name]}
    )
    dataarray.attrs = {}
    return dataarray
