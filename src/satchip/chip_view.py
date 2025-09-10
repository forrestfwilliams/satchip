import argparse
from collections import namedtuple
from itertools import product
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from satchip.utils import load_chip


Index = namedtuple('Index', ['sample', 'time'])


def view_chip(label_path: Path, band: str, idx: int = 0) -> None:
    chip = load_chip(label_path)
    band_names = list(chip['band'].values)
    if band not in band_names:
        raise ValueError(f'Band {band} not found in chip. Available bands: {", ".join(band_names)}')
    da = chip['data'].sel(band=band, platform=chip['platform'].values[0]).drop_vars(['platform', 'band'])

    indexes = [Index(s, t) for s, t in product(da.sample.values, da.time.values)]

    # Initial plot
    start_index = indexes[idx]
    img = da.sel(sample=start_index.sample, time=start_index.time).plot.imshow(
        add_colorbar=True, cmap='gray', figsize=(10, 10)
    )
    assert img.colorbar is not None
    img.colorbar.set_label('')
    ax = img.axes
    date = start_index.time.astype('datetime64[ms]').astype(object).strftime('%Y-%m-%d')
    title = f'Date: {date} | Band: {band} | Sample: {da.sample.values[idx]}'
    ax.set_title(title)
    ax.set_aspect('equal')
    fig = ax.figure

    # Slider axis
    slider_ax = fig.add_axes([0.2, 0.02, 0.6, 0.03])  # type: ignore
    slider = Slider(slider_ax, 'Index', 0, len(indexes) - 1, valinit=idx, valstep=1)

    def update(val: int) -> None:
        assert val is not None
        index = indexes[int(slider.val)]
        img.set_data(da.sel(sample=index.sample, time=index.time).values)
        date = index.time.astype('datetime64[ms]').astype(object).strftime('%Y-%m-%d')
        title = f'Date: {date} | Band: {band} | Sample: {index.sample}'
        ax.set_title(title)
        fig.canvas.draw_idle()

    slider.on_changed(update)  # type: ignore
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('chippath', type=Path, help='Path to the label image')
    parser.add_argument('band', type=str, help='Band to view')
    parser.add_argument('--idx', type=int, default=0, help='Index of default sample to view')
    args = parser.parse_args()
    view_chip(args.chippath, args.band, args.idx)


if __name__ == '__main__':
    main()
