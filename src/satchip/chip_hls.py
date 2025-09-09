from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import numpy as np
import rioxarray
import shapely
import xarray as xr
from earthaccess.results import DataGranule

from satchip.chip_xr_base import create_template_da
from satchip.terra_mind_grid import TerraMindChip


HLS_L_BANDS = {
    'B01': 'COASTAL',
    'B02': 'BLUE',
    'B03': 'GREEN',
    'B04': 'RED',
    'B05': 'NIR08',
    'B06': 'SWIR16',
    'B07': 'SWIR22',
}
HLS_S_BANDS = {
    'B01': 'COASTAL',
    'B02': 'BLUE',
    'B03': 'GREEN',
    'B04': 'RED',
    'B8A': 'NIR08',
    'B11': 'SWIR16',
    'B12': 'SWIR22',
}
BAND_SETS = {'L30': HLS_L_BANDS, 'S30': HLS_S_BANDS}


def get_pct_intersect(umm: dict, roi: shapely.geometry.Polygon) -> float:
    points = umm['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]['Boundary']['Points']
    coords = [(pt['Longitude'], pt['Latitude']) for pt in points]
    image_roi = shapely.geometry.Polygon(coords)
    return roi.intersection(image_roi).area / roi.area


def get_date(umm: dict) -> datetime:
    date_fmt = '%Y-%m-%dT%H:%M:%S'
    date = [x['Values'][0].split('.')[0] for x in umm['AdditionalAttributes'] if x['Name'] == 'SENSING_TIME'][0]
    return datetime.strptime(date, date_fmt)


def get_product_id(umm: dict) -> str:
    return [x['Identifier'] for x in umm['DataGranule']['Identifiers'] if x['IdentifierType'] == 'ProducerGranuleId'][0]


def get_scenes(
    items: list[DataGranule], roi: shapely.geometry.Polygon, max_cloud_pct: int, strategy: str, scratch_dir: Path
) -> list[DataGranule]:
    """Returns the best HLS scene from the given list of items.
    The best scene is defined as the earliest scene with the largest intersection with the roi and
    less than or equal to the max_cloud_pct.

    Args:
        items: List of HLS earthaccess result items.
        roi: Region of interest polygon.
        max_cloud_pct: Maximum percent of bad pixels allowed in the scene.
        strategy: Strategy to use when selecting data.
        scratch_dir: Directory to store downloaded files.

    Returns:
        The best HLS item.
    """
    overlapping_items = [x for x in items if get_pct_intersect(x['umm'], roi) > 95]
    best_first = sorted(overlapping_items, key=lambda x: (-get_pct_intersect(x['umm'], roi), get_date(x['umm'])))
    valid_scenes = []
    for item in best_first:
        product_id = get_product_id(item['umm'])
        n_products = len(list(scratch_dir.glob(f'{product_id}*')))
        if n_products < 15:
            earthaccess.download([item], scratch_dir, pqdm_kwargs={'disable': True})
        fmask_path = scratch_dir / f'{product_id}.v2.0.FMask.tif'
        assert fmask_path.exists(), f'File not found: {fmask_path}'
        qual_da = rioxarray.open_rasterio(fmask_path).rio.clip_box(*roi.bounds, crs='EPSG:4326')  # type: ignore
        bit_masks = np.unpackbits(qual_da.data[0][..., np.newaxis], axis=-1)
        # Looks for a 1 in the 4th, 6th and 7th bit of the Fmask (reverse order). See table 9 and appendix A of:
        # https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
        bad_pixels = (bit_masks[..., 4] == 1) | (bit_masks[..., 6] == 1) | (bit_masks[..., 7] == 1)
        pct_bad = int(np.round(100 * np.sum(bad_pixels) / bad_pixels.size))
        if pct_bad <= max_cloud_pct:
            if strategy == 'BEST':
                return [item]
            else:
                valid_scenes.append(item)
    if strategy == 'ALL':
        return valid_scenes
    raise ValueError(f'No HLS scenes found with <= {max_cloud_pct}% cloud cover for chip.')


def get_hls_data(chip: TerraMindChip, scratch_dir: Path, opts: dict) -> xr.DataArray:
    """Returns XArray DataArray of a Harmonized Landsat Sentinel-2 image for the given bounds and
    closest collection after date.
    """
    date_start = opts['date_start']
    date_end = opts['date_end'] + timedelta(days=1)  # inclusive end
    earthaccess.login()
    results = earthaccess.search_data(
        short_name=['HLSL30', 'HLSS30'], bounding_box=chip.bounds, temporal=(date_start, date_end)
    )
    assert len(results) > 0, f'No HLS scenes found for chip {chip.name} between {date_start} and {date_end}.'
    roi = shapely.box(*chip.bounds)
    roi_buffered = roi.buffer(0.01)
    max_cloud_pct = opts.get('max_cloud_pct', 100)
    strategy = opts.get('strategy', 'BEST').upper()
    scenes = get_scenes(results, roi, max_cloud_pct, strategy, scratch_dir)
    das = []
    template = create_template_da(chip)
    for scene in scenes:
        product_id = get_product_id(scene['umm'])
        bands = BAND_SETS[product_id.split('.')[1]]
        for band in bands:
            image_path = scratch_dir / f'{product_id}.v2.0.{band}.tif'
            da = rioxarray.open_rasterio(image_path).rio.clip_box(*roi_buffered.bounds, crs='EPSG:4326')  # type: ignore
            da_reproj = da.rio.reproject_match(template)
            da_reproj['band'] = [bands[band]]
            da_reproj = da_reproj.expand_dims({'time': [get_date(scene['umm']).replace(tzinfo=None)]})
            da_reproj['x'] = np.arange(0, chip.ncol)
            da_reproj['y'] = np.arange(0, chip.nrow)
            da_reproj.attrs = {}
            das.append(da_reproj)
    dataarray = xr.combine_by_coords(das, join='override').drop_vars('spatial_ref')
    assert isinstance(dataarray, xr.DataArray)
    dataarray = dataarray.expand_dims({'sample': [chip.name], 'platform': ['HLS']})
    dataarray.attrs = {}
    return dataarray
