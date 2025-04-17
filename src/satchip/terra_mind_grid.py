from itertools import product

import pyproj
from rasterio import Affine

from satchip.major_tom_grid import MajorTomGrid
from satchip.utils import get_epsg4326_bbox, get_epsg4326_point


TERRA_MIND_CHIP_SIZE = 264


class Chip:
    def __init__(
        self, name: str, minx: float, maxy: float, xres: float, yres: float, nrow: int, ncol: int, epsg: int
    ) -> None:
        self.name = name
        self.minx = minx
        self.maxy = maxy
        self.xres = xres
        self.yres = yres
        self.nrow = nrow
        self.ncol = ncol
        self.epsg = epsg
        self.center = self.get_center()
        self.bounds = self.get_bounds()
        self.gdal_transform = self.get_gdal_transform()
        self.rio_transform = Affine.from_gdal(*self.gdal_transform)

    def get_center(self) -> tuple:
        center_x = self.minx + (self.ncol * self.xres) / 2
        center_y = self.maxy + (self.nrow * self.yres) / 2
        return get_epsg4326_point(center_x, center_y, self.epsg)

    def get_bounds(self) -> tuple:
        minx = self.minx
        maxy = self.maxy
        maxx = minx + (self.ncol * self.xres)
        miny = maxy + (self.nrow * self.yres)
        return get_epsg4326_bbox((minx, miny, maxx, maxy), self.epsg, buffer=0)

    def get_gdal_transform(self) -> tuple:
        return (self.minx, self.xres, 0.0, self.maxy, 0.0, self.yres)

    def __repr__(self) -> str:
        return f'{self.name}: {self.gdal_transform}'


class MajorTomChip(Chip):
    def __init__(self, name: str, minx: float, maxy: float, epsg: int) -> None:
        super().__init__(name=name, minx=minx, maxy=maxy, xres=10, yres=-10, nrow=1068, ncol=1068, epsg=epsg)


class TerraMindChip(Chip):
    def __init__(self, name: str, minx: float, maxy: float, epsg: int) -> None:
        super().__init__(
            name=name,
            minx=minx,
            maxy=maxy,
            xres=10,
            yres=-10,
            nrow=TERRA_MIND_CHIP_SIZE,
            ncol=TERRA_MIND_CHIP_SIZE,
            epsg=epsg,
        )


class TerraMindGrid:
    def __init__(self, latitude_range: tuple, longitude_range: tuple) -> None:
        self.major_tom_grid = MajorTomGrid(
            dist=10, latitude_range=latitude_range, longitude_range=longitude_range, utm_definition='bottomleft'
        )
        self.major_tom_chips = self.get_major_tom_chips()
        self.terra_mind_chips = self.get_terra_mind_chips()
        self.transform_groups = self.get_transform_groups()

    def get_major_tom_chips(self) -> list:
        if self.major_tom_grid.dist != 10:
            raise ValueError('This function is only valid for a grid distance of 10 km.')

        major_tom_chips = []
        for _, point in self.major_tom_grid.points.iterrows():
            utm_epsg = int(point['utm_zone'])
            latlng = pyproj.CRS('EPSG:4326')
            utm = pyproj.CRS(f'EPSG:{utm_epsg}')
            latlng2utm = pyproj.Transformer.from_crs(latlng, utm, always_xy=True)
            minx, miny = latlng2utm.transform(point.geometry.x, point.geometry.y)
            maxy = miny + 1068 * 10  # 1068 pixel chip at 10m cell size
            major_tom_chips.append(MajorTomChip(name=point['name'], minx=minx, maxy=maxy, epsg=utm_epsg))
        return major_tom_chips

    @staticmethod
    def get_terra_mind_chips_for_major_tom_chip(major_tom_chip: MajorTomChip) -> list:
        terra_mind_chips = []
        for col, row in product(range(0, 4), range(0, 4)):
            name = f'{major_tom_chip.name}_{col}_{row}'
            maxy = major_tom_chip.maxy + (row * TERRA_MIND_CHIP_SIZE * major_tom_chip.yres)
            minx = major_tom_chip.minx + (col * TERRA_MIND_CHIP_SIZE * major_tom_chip.xres)
            terra_mind_chip = TerraMindChip(name=name, minx=minx, maxy=maxy, epsg=major_tom_chip.epsg)
            terra_mind_chips.append(terra_mind_chip)
        return terra_mind_chips

    def get_terra_mind_chips(self) -> list:
        terra_mind_chips = []
        for major_tom_chip in self.major_tom_chips:
            terra_mind_chips += self.get_terra_mind_chips_for_major_tom_chip(major_tom_chip)
        return terra_mind_chips

    def get_transform_groups(self) -> dict:
        transform_groups = {}
        epsgs = set([chip.epsg for chip in self.terra_mind_chips])
        for epsg in epsgs:
            minx = min([chip.minx for chip in self.terra_mind_chips if chip.epsg == epsg])
            maxy = max([chip.maxy for chip in self.terra_mind_chips if chip.epsg == epsg])
            gdal_transform = (minx, 10, 0, maxy, 0, -10)
            transform_groups[epsg] = Affine(*gdal_transform)
        return transform_groups
