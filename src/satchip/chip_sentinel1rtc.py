from datetime import datetime, timedelta
from pathlib import Path

import asf_search as search
import shapely
import xarray as xr
from asf_search import constants
from hyp3_sdk import HyP3
from hyp3_sdk.utils import extract_zipped_product

from satchip.terra_mind_grid import TerraMindChip


def sort_products(product: search.ASFProduct, roi: shapely.geometry.Polygon) -> tuple:
    footprint = shapely.geometry.shape(product.geometry)
    intersection = int(100 * roi.intersection(footprint).area / roi.area) * -1
    date = product.properties['startTime']
    return intersection, date


def get_hyp3_rtc(scene_name: str, scratch_dir: Path) -> Path:
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
        job.download_files(local_dir=scratch_dir)
        extract_zipped_product(output_zip)

    return output_dir


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
    rtc_path = get_hyp3_rtc(scene_name, scratch_dir)
    breakpoint()
    return None
