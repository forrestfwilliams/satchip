from satchip import grid


def test_get_utm_zone_for_latlng():
    assert grid.get_utm_zone_from_latlng([-1, -174.34]) == 32701
    assert grid.get_utm_zone_from_latlng([48, -4]) == 32630
    assert grid.get_utm_zone_from_latlng([78, 13]) == 32633
    assert grid.get_utm_zone_from_latlng([-34, 19.7]) == 32734
    assert grid.get_utm_zone_from_latlng([-36, 175.7]) == 32760
