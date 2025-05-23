import math
import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


class MajorTomGrid:
    RADIUS_EQUATOR = 6378.137  # km

    def __init__(
        self,
        dist: int,
        latitude_range: tuple = (-85, 85),
        longitude_range: tuple = (-180, 180),
        utm_definition: str = 'bottomleft',
    ) -> None:
        self.dist = dist
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
        self.utm_definition = utm_definition
        self.rows, self.lats = self.get_rows()
        self.points, self.points_by_row = self.get_points()

    def get_rows(self) -> tuple:
        # Define set of latitudes to use, based on the grid distance
        arc_pole_to_pole = math.pi * self.RADIUS_EQUATOR
        num_divisions_in_hemisphere = math.ceil(arc_pole_to_pole / self.dist)

        latitudes = np.linspace(-90, 90, num_divisions_in_hemisphere + 1)[:-1]
        latitudes = np.mod(latitudes, 180) - 90

        # order should be from south to north
        latitudes = np.sort(latitudes)

        zeroth_row = np.searchsorted(latitudes, 0)

        # From 0U-NU and 1D-ND
        rows: list[Any] = [None] * len(latitudes)
        rows[zeroth_row:] = [f'{i}U' for i in range(len(latitudes) - zeroth_row)]
        rows[:zeroth_row] = [f'{abs(i - zeroth_row)}D' for i in range(zeroth_row)]

        # bound to range
        idxs = (latitudes >= self.latitude_range[0]) * (latitudes <= self.latitude_range[1])
        rows, latitudes = np.array(rows), np.array(latitudes)
        rows, latitudes = rows[idxs], latitudes[idxs]

        return rows, latitudes

    def get_circumference_at_latitude(self, lat: float) -> float:
        # Circumference of the cross-section of a sphere at a given latitude

        radius_at_lat = self.RADIUS_EQUATOR * math.cos(lat * math.pi / 180)
        circumference = 2 * math.pi * radius_at_lat

        return circumference

    def subdivide_circumference(self, lat: float, return_cols: bool = False) -> np.ndarray:
        # Provide a list of longitudes that subdivide the circumference of the earth at a given latitude
        # into equal parts as close as possible to dist

        circumference = self.get_circumference_at_latitude(lat)
        num_divisions = math.ceil(circumference / self.dist)
        longitudes = np.linspace(-180, 180, num_divisions + 1)[:-1]
        longitudes = np.mod(longitudes, 360) - 180
        longitudes = np.sort(longitudes)

        if return_cols:
            cols: list[Any] = [None] * len(longitudes)
            zeroth_idx = np.where(longitudes == 0)[0][0]
            cols[zeroth_idx:] = [f'{i}R' for i in range(len(longitudes) - zeroth_idx)]
            cols[:zeroth_idx] = [f'{abs(i - zeroth_idx)}L' for i in range(zeroth_idx)]
            return np.array(cols), np.array(longitudes)

        return np.array(longitudes)

    def get_points(self) -> tuple:
        r_idx = 0
        points_by_row = [None] * len(self.rows)
        for r, lat in zip(self.rows, self.lats):
            (
                point_names,
                grid_row_names,
                grid_col_names,
                grid_row_idx,
                grid_col_idx,
                grid_lats,
                grid_lons,
                utm_zones,
                epsgs,
            ) = [], [], [], [], [], [], [], [], []
            cols, lons = self.subdivide_circumference(lat, return_cols=True)

            cols, lons = self.filter_longitude(cols, lons)
            c_idx = 0
            for c, lon in zip(cols, lons):
                point_names.append(f'{r}_{c}')
                grid_row_names.append(r)
                grid_col_names.append(c)
                grid_row_idx.append(r_idx)
                grid_col_idx.append(c_idx)
                grid_lats.append(lat)
                grid_lons.append(lon)
                if self.utm_definition == 'bottomleft':
                    utm_zones.append(get_utm_zone_from_latlng([lat, lon]))
                elif self.utm_definition == 'center':
                    center_lat = lat + (1000 * self.dist / 2) / 111_120
                    center_lon = lon + (1000 * self.dist / 2) / (111_120 * math.cos(center_lat * math.pi / 180))
                    utm_zones.append(get_utm_zone_from_latlng([center_lat, center_lon]))
                else:
                    raise ValueError(f'Invalid utm_definition {self.utm_definition}')
                epsgs.append(f'EPSG:{utm_zones[-1]}')

                c_idx += 1
            points_by_row[r_idx] = gpd.GeoDataFrame(
                {
                    'name': point_names,
                    'row': grid_row_names,
                    'col': grid_col_names,
                    'row_idx': grid_row_idx,
                    'col_idx': grid_col_idx,
                    'utm_zone': utm_zones,
                    'epsg': epsgs,
                },
                geometry=gpd.points_from_xy(grid_lons, grid_lats),
            )
            r_idx += 1
        points = gpd.GeoDataFrame(pd.concat(points_by_row))
        # points.reset_index(inplace=True,drop=True)
        return points, points_by_row

    def group_points_by_row(self) -> gpd.GeoDataFrame:
        # Make list of different gdfs for each row
        points_by_row = [None] * len(self.rows)
        for i, row in enumerate(self.rows):
            points_by_row[i] = self.points[self.points.row == row]
        return points_by_row

    def filter_longitude(self, cols: tuple, lons: tuple) -> tuple:
        idxs = (lons >= self.longitude_range[0]) * (lons <= self.longitude_range[1])
        cols, lons = cols[idxs], lons[idxs]
        return cols, lons

    def latlon2rowcol(self, lats: tuple, lons: tuple, return_idx: bool = False, integer: bool = False) -> list:
        # Always take bottom left corner of grid cell
        rows = np.searchsorted(self.lats, lats) - 1

        # Get the possible points of the grid cells at the given latitude
        possible_points = [self.points_by_row[row] for row in rows]

        # For each point, find the rightmost point that is still to the left of the given longitude
        cols = [
            poss_points.iloc[np.searchsorted(poss_points.geometry.x, lon) - 1].col
            for poss_points, lon in zip(possible_points, lons)
        ]
        rows = self.rows[rows].tolist()

        outputs = [list(rows), list(cols)]
        if return_idx:
            # Get the table index for self.points with each row,col pair in rows, cols
            idx = [
                self.points[(self.points.row == row) & (self.points.col == col)].index.values[0]
                for row, col in zip(rows, cols)
            ]
            outputs.append(idx)

        # return raw numbers
        if integer:
            outputs[0] = [int(el[:-1]) if el[-1] == 'U' else -int(el[:-1]) for el in outputs[0]]
            outputs[1] = [int(el[:-1]) if el[-1] == 'R' else -int(el[:-1]) for el in outputs[1]]

        return outputs

    def rowcol2latlon(self, rows: tuple, cols: tuple) -> tuple:
        point_geoms = [
            self.points.loc[(self.points.row == row) & (self.points.col == col), 'geometry'].values[0]
            for row, col in zip(rows, cols)
        ]
        lats = [point.y for point in point_geoms]
        lons = [point.x for point in point_geoms]
        return lats, lons

    def get_bounded_footprint(self, point: gpd.GeoDataFrame, buffer_ratio: float = 0) -> Polygon:
        # Gets the polygon footprint of the grid cell for a given point, bounded by the other grid points' cells.
        # Grid point defined as bottom-left corner of polygon. Buffer ratio is the ratio of the grid cell's width/height to buffer by.

        bottom, left = point.geometry.y, point.geometry.x
        row_idx = point.row_idx
        col_idx = point.col_idx
        next_row_idx = row_idx + 1
        next_col_idx = col_idx + 1

        if next_row_idx >= len(self.lats):  # If at top row, use difference between top and second-to-top row for height
            height = self.lats[row_idx] - self.lats[row_idx - 1]
            top = self.lats[row_idx] + height
        else:
            top = self.lats[next_row_idx]

        max_col = len(self.points_by_row[row_idx].col_idx) - 1
        if (
            next_col_idx > max_col
        ):  # If at rightmost column, use difference between rightmost and second-to-rightmost column for width
            width = (
                self.points_by_row[row_idx].iloc[col_idx].geometry.x
                - self.points_by_row[row_idx].iloc[col_idx - 1].geometry.x
            )
            right = self.points_by_row[row_idx].iloc[col_idx].geometry.x + width
        else:
            right = self.points_by_row[row_idx].iloc[next_col_idx].geometry.x

        # Buffer the polygon by the ratio of the grid cell's width/height
        width = right - left
        height = top - bottom

        buffer_horizontal = width * buffer_ratio
        buffer_vertical = height * buffer_ratio

        new_left = left - buffer_horizontal
        new_right = right + buffer_horizontal

        new_bottom = bottom - buffer_vertical
        new_top = top + buffer_vertical

        bbox = Polygon([(new_left, new_bottom), (new_left, new_top), (new_right, new_top), (new_right, new_bottom)])

        return bbox


def get_utm_zone_from_latlng(latlng: list) -> int:
    assert isinstance(latlng, list | tuple), 'latlng must be in the form of a list or tuple.'

    longitude = latlng[1]
    latitude = latlng[0]

    zone_number = (math.floor((longitude + 180) / 6)) % 60 + 1

    # Special zones for Svalbard and Norway
    if latitude >= 56.0 and latitude < 64.0 and longitude >= 3.0 and longitude < 12.0:
        zone_number = 32
    elif latitude >= 72.0 and latitude < 84.0:
        if longitude >= 0.0 and longitude < 9.0:
            zone_number = 31
        elif longitude >= 9.0 and longitude < 21.0:
            zone_number = 33
        elif longitude >= 21.0 and longitude < 33.0:
            zone_number = 35
        elif longitude >= 33.0 and longitude < 42.0:
            zone_number = 37

    # Determine the hemisphere and construct the EPSG code
    if latitude < 0:
        epsg_code = f'327{zone_number:02d}'
    else:
        epsg_code = f'326{zone_number:02d}'
    if not re.match(r'32[6-7](0[1-9]|[1-5][0-9]|60)', epsg_code):
        print(f'latlng: {latlng}, epsg_code: {epsg_code}')
        raise ValueError('out of bound latlng resulted in incorrect EPSG code for the point')

    return int(epsg_code)
