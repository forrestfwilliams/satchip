# SatChip

A package for satellite image AI data prep. This package "chips" data labels and satellite imagery into 264x264 image arrays following the TerraMind extension of the MajorTom specification.

## Using
`SatChip` relies on a two-step process; chip your label train data inputs, then create corresponding chips for different remote sensing data sources.

### Step 1: Chip labels
The `chiplabel` CLI tool takes a GDAL-compatible image, a collection date, and an optional output directory as input using the following format:

```bash
chiplabel PATH/TO/LABELS.tif DATE(UTC FORMAT) --outdir OUTPUT_DIR
```
For example:
```bash
chiplabel LA_damage_20250113_v0.tif 2024-01-01T01:01:01 --outdir chips
```
This will produce an output zipped Zarr store label dataset with the name `{LABELS}.zarr.zip` in the specified output directory (`--outdir`).
This file will be the input to the remote sensing data chipping step.

For more information on usage see `chiplabel --help`

### Step 2: Chip remote sensing data
The `chipdata` CLI tool takes a label zipped Zarr store, a dataset name, and an optional output directory as input using the following format:
```bash
chipdata PATH/TO/LABELS.zarr.zip DATASET --outdir OUTPUT_DIR
```
For example:
```bash
chipdata LA_damage_20250113_v0.zarr.zip S2L2A --outdir chips
```
Similarly to step 1, this will produce an output zipped Zarr store that contains chipped data for your chosen dataset with the name `{LABELS}_{DATASET}.zarr.zip`.

Currently support datasets include:
- `S2L2A`: Sentinel-2 L2A data sourced from the [Sentinel-2 AWS Open Data Archive](https://registry.opendata.aws/sentinel-2/)
- `S1RTC`: Sentinel-1 Radiometric Terrain Corrected (RTC) data created using [ASF's HyP3 on-demand platform](https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/)

## License
`SatChip` is licensed under the BSD-3-Clause open source license. See the LICENSE file for more details.

## Contributing
Contributions to the `SatChip` are welcome! If you would like to contribute, please submit a pull request on the GitHub repository.
