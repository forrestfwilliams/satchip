import numpy as np
import xarray as xr

from satchip.terra_mind_grid import TerraMindChip


def create_template_da(chip: TerraMindChip) -> xr.DataArray:
    """Create a template DataArray with the same dimensions and transform as a label chip."""
    x = np.arange(chip.nrow) * chip.gdal_transform[1] + chip.gdal_transform[0] + chip.gdal_transform[1] / 2
    y = np.arange(chip.ncol) * chip.gdal_transform[5] + chip.gdal_transform[3] + chip.gdal_transform[5] / 2
    template = xr.DataArray(np.zeros((chip.ncol, chip.nrow)), dims=('y', 'x'), coords={'y': y, 'x': x})
    template.rio.write_crs(f'EPSG:{chip.epsg}', inplace=True)
    template.rio.write_transform(chip.rio_transform, inplace=True)
    return template
