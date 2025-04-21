import numpy as np
import pytest

from satchip.terra_mind_grid import TerraMindGrid


@pytest.mark.parametrize(
    'tm_name, point',
    [
        ('374U_897R_3_2', [96.80658, 33.62825]),
        ('735U_418L_0_1', [-92.35263, 66.07510]),
        ('611U_462L_2_1', [-72.02323, 54.93515]),
    ],
)
def test_terra_mind_grid(tm_name, point):
    mt_name = f'{tm_name.split("_")[0]}_{tm_name.split("_")[1]}'
    tmp_grid = TerraMindGrid([np.floor(point[1]), np.ceil(point[1])], [np.floor(point[0]), np.ceil(point[0])])
    # mt_chip = [x for x in tmp_grid.major_tom_chips if x.name == mt_name][0]
    chips = [x for x in tmp_grid.terra_mind_chips if x.name.startswith(mt_name)]
    in_lon = [x for x in chips if x.bounds[0] < point[0] < x.bounds[2]]
    in_lon_lat = [x for x in in_lon if x.bounds[1] < point[1] < x.bounds[3]]
    assert len(in_lon_lat) == 1
    assert in_lon_lat[0].name == tm_name
