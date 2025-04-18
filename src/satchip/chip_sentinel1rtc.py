from datetime import datetime, timedelta
from pathlib import Path

import asf_search as search
import numpy as np
import rioxarray
import shapely
import xarray as xr
from asf_search import constants
from hyp3_sdk import HyP3
from hyp3_sdk.util import extract_zipped_product

from satchip.chip_xr_base import create_template_da
from satchip.terra_mind_grid import TerraMindChip


def sort_products(product: search.ASFProduct, roi: shapely.geometry.Polygon) -> tuple:
    footprint = shapely.geometry.shape(product.geometry)
    intersection = int(100 * roi.intersection(footprint).area / roi.area) * -1
    date = product.properties['startTime']
    return intersection, date


def get_hyp3_rtc(scene_name: str, scratch_dir: Path) -> tuple[Path, Path]:
    hyp3 = HyP3()
    jobs = [j for j in hyp3.find_jobs(job_type='RTC_GAMMA') if not j.failed() and not j.expired()]
    jobs = [j for j in jobs if j.job_parameters['granules'] == [scene_name]]
    jobs = [j for j in jobs if j.job_parameters['radiometry'] == 'gamma0']
    jobs = [j for j in jobs if j.job_parameters['resolution'] == 20]

    if len(jobs) == 0:
        job = hyp3.submit_rtc_job(scene_name, radiometry='gamma0', resolution=20)
    else:
        job = jobs[0]

    if not job.succeeded():
        hyp3.watch(job)

    output_path = scratch_dir / jobs[0].to_dict()['files'][0]['filename']
    output_dir = output_path.with_suffix('')
    output_zip = output_path.with_suffix('.zip')
    if not output_dir.exists():
        job.download_files(location=scratch_dir)
        extract_zipped_product(output_zip)
    vv_path = list(output_dir.glob('*_VV.tif'))[0]
    vh_path = list(output_dir.glob('*_VH.tif'))[0]
    return vv_path, vh_path


def get_s1rtc_data(chip: TerraMindChip, date: datetime, scratch_dir: Path) -> xr.DataArray:
    roi = shapely.box(*chip.bounds)
    search_results = search.geo_search(
        intersectsWith=roi.wkt,
        start=date,
        end=date + timedelta(weeks=2),
        beamMode=constants.BEAMMODE.IW,
        polarization=constants.POLARIZATION.VV_VH,
        platform=constants.PLATFORM.SENTINEL1,
        processingLevel=constants.PRODUCT_TYPE.SLC,
    )
    if len(search_results) == 0:
        raise ValueError(f'No products found for chip {chip.name} on {date}')
    product = sorted(list(search_results), key=lambda x: sort_products(x, roi))[0]
    scene_name = product.properties['sceneName']
    vv_path, vh_path = get_hyp3_rtc(scene_name, scratch_dir)
    das = []
    template = create_template_da(chip)
    for band_name, image_path in zip(['VV', 'VH'], [vv_path, vh_path]):
        da = rioxarray.open_rasterio(image_path).rio.clip_box(*roi.buffer(0.1).bounds, crs='EPSG:4326')
        da['band'] = [band_name]
        da_reproj = da.rio.reproject_match(template)
        das.append(da_reproj)
    dataarray = xr.concat(das, dim='band').drop_vars('spatial_ref')
    dataarray['x'] = np.arange(0, chip.ncol)
    dataarray['y'] = np.arange(0, chip.nrow)
    time = datetime.strptime(product.properties['startTime'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)
    dataarray = dataarray.expand_dims({'time': [time], 'sample': [chip.name]})
    dataarray.attrs = {}
    return dataarray
