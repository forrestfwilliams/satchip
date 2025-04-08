import xarray as xr
import zarr
from pyproj import CRS, Transformer


def get_epsg4326_point(x: float, y: float, in_epsg: int) -> list:
    if in_epsg == 4326:
        return x, y
    in_crs = CRS.from_epsg(in_epsg)
    out_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)
    newx, newy = transformer.transform(x, y)
    return round(newx, 5), round(newy, 5)


def get_epsg4326_bbox(bounds: list, in_epsg: int, buffer: float = 0.1) -> list:
    minx, miny = get_epsg4326_point(bounds[0], bounds[1], in_epsg)
    maxx, maxy = get_epsg4326_point(bounds[2], bounds[3], in_epsg)
    bbox = minx - buffer, miny - buffer, maxx + buffer, maxy + buffer
    return list(bbox)


def get_overall_bounds(bounds: list) -> list:
    minx = min([b[0] for b in bounds])
    miny = min([b[1] for b in bounds])
    maxx = max([b[2] for b in bounds])
    maxy = max([b[3] for b in bounds])
    return [minx, miny, maxx, maxy]


def save_chip(dataset: xr.Dataset, save_path: str) -> None:
    """Save a zipped zarr archive"""
    store = zarr.storage.ZipStore(save_path, mode='w')
    dataset.to_zarr(store)


def load_chip(label_path: str) -> xr.Dataset:
    """Load a zipped zarr archive"""
    store = zarr.storage.ZipStore(label_path, read_only=True)
    dataset = xr.open_zarr(store)
    return dataset
