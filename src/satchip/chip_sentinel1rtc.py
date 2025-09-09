from datetime import datetime, timedelta
from pathlib import Path

import asf_search as search
import numpy as np
import rioxarray
import shapely
import xarray as xr
from asf_search import ASFSearchResults, S1Product, constants
from hyp3_sdk import Batch, HyP3, Job
from hyp3_sdk.util import extract_zipped_product

from satchip.chip_xr_base import create_template_da
from satchip.terra_mind_grid import TerraMindChip


def get_pct_intersect(product: S1Product, roi: shapely.geometry.Polygon) -> int:
    footprint = shapely.geometry.shape(product.geometry)
    intersection = int(np.round(100 * roi.intersection(footprint).area / roi.area))
    return intersection


def download_hyp3_rtc(job: Job, scratch_dir: Path) -> tuple[Path, Path]:
    output_path = scratch_dir / job.to_dict()['files'][0]['filename']
    output_dir = output_path.with_suffix('')
    output_zip = output_path.with_suffix('.zip')
    if not output_dir.exists():
        job.download_files(location=scratch_dir)
        extract_zipped_product(output_zip)
    vv_path = list(output_dir.glob('*_VV.tif'))[0]
    vh_path = list(output_dir.glob('*_VH.tif'))[0]
    return vv_path, vh_path


def get_hyp3_rtcs(items: ASFSearchResults, roi: shapely.geometry.Polygon, strategy: str, scratch_dir: Path) -> list:
    valid_items = [item for item in items if get_pct_intersect(item, roi) > 95]
    valid_items = sorted(items, key=lambda x: (-get_pct_intersect(x, roi), x.properties['startTime']))
    if strategy == 'BEST':
        valid_items = valid_items[:1]
    hyp3 = HyP3()
    old_jobs = [j for j in hyp3.find_jobs(job_type='RTC_GAMMA') if not j.failed() and not j.expired()]
    old_jobs = [j for j in old_jobs if j.job_parameters['radiometry'] == 'gamma0']  # type: ignore
    old_jobs = [j for j in old_jobs if j.job_parameters['resolution'] == 20]  # type: ignore
    jobs = []
    for item in valid_items:
        scene_name = item.properties['sceneName']
        matching_jobs = [j for j in old_jobs if j.job_parameters['granules'] == [scene_name]]  # type: ignore
        if len(matching_jobs) == 0:
            job = hyp3.submit_rtc_job(scene_name, radiometry='gamma0', resolution=20)
            jobs.append(job)
        else:
            jobs.append(matching_jobs[0])
    jobs = Batch(jobs)
    hyp3.watch(jobs)
    assert all([j.succeeded() for j in jobs]), 'One or more HyP3 jobs failed'
    paths = [download_hyp3_rtc(job, scratch_dir) for job in jobs]
    return paths


def get_s1rtc_data(chip: TerraMindChip, scratch_dir: Path, opts: dict) -> xr.DataArray:
    date_start = opts['date_start']
    date_end = opts['date_end'] + timedelta(days=1)  # inclusive end
    roi = shapely.box(*chip.bounds)
    search_results = search.geo_search(
        intersectsWith=roi.wkt,
        start=date_start,
        end=date_end,
        beamMode=constants.BEAMMODE.IW,
        polarization=constants.POLARIZATION.VV_VH,
        platform=constants.PLATFORM.SENTINEL1,
        processingLevel=constants.PRODUCT_TYPE.SLC,
    )
    if len(search_results) == 0:
        raise ValueError(f'No products found for chip {chip.name} in date range {date_start} to {date_end}')
    strategy = opts.get('strategy', 'BEST').upper()
    image_sets = get_hyp3_rtcs(search_results, roi, strategy, scratch_dir)
    das = []
    template = create_template_da(chip)
    for image_set in image_sets:
        for band_name, image_path in zip(['VV', 'VH'], image_set):
            da = rioxarray.open_rasterio(image_path).rio.clip_box(*roi.buffer(0.1).bounds, crs='EPSG:4326')  # type: ignore
            da_reproj = da.rio.reproject_match(template)
            da_reproj['band'] = [band_name]
            image_time = datetime.strptime(image_path.name.split('_')[2], '%Y%m%dT%H%M%S')
            da_reproj = da_reproj.expand_dims({'time': [image_time]})
            da_reproj['x'] = np.arange(0, chip.ncol)
            da_reproj['y'] = np.arange(0, chip.nrow)
            da_reproj.attrs = {}
            das.append(da_reproj)
    dataarray = xr.combine_by_coords(das, join='override').drop_vars('spatial_ref')
    assert isinstance(dataarray, xr.DataArray)
    dataarray = dataarray.expand_dims({'sample': [chip.name], 'platform': ['S1RTC']})
    return dataarray
