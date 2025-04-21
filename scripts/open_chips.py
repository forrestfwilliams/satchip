import argparse
from pathlib import Path

import xarray as xr


def open_chips(input_path: Path) -> xr.Dataset:
    ds = xr.open_zarr(input_path)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description='Open a chip Zarr Zip Store')
    parser.add_argument('input', type=str, help='Path to the input Zarr Zip Store')
    args = parser.parse_args()
    args.input = Path(args.input)

    open_chips(args.input)


if __name__ == '__main__':
    main()
