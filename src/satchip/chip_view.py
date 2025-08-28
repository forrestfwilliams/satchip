import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from satchip.utils import load_chip


def view_chip(label_path: Path, band: str, idx: int = 0) -> None:
    chip = load_chip(label_path)
    band_names = list(chip['band'].values)
    if band not in band_names:
        raise ValueError(f'Band {band} not found in chip. Available bands: {", ".join(band_names)}')
    da = chip.sel(band=band).bands.squeeze()

    # Initial plot
    img = da.isel(sample=idx).plot.imshow(add_colorbar=True, cmap='gray', figsize=(10, 10))
    img.colorbar.set_label('')
    ax = img.axes
    date = da.time.values.astype('datetime64[ms]').astype(object).strftime('%Y-%m-%d')
    title = f'Date: {date} | Band: {band} | Sample: {da.sample.values[idx]}'
    ax.set_title(title)
    ax.set_aspect('equal')
    fig = ax.figure

    # Slider axis
    slider_ax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, 'Index', 0, len(da.sample) - 1, valinit=idx, valstep=1)

    def update(val: int) -> None:
        sidx = int(slider.val)
        img.set_data(da.isel(sample=sidx).values)
        title = f'Date: {date} | Band: {band} | Sample: {da.sample.values[sidx]}'
        ax.set_title(title)
        fig.canvas.draw_idle()

    slider.on_changed(update)
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
