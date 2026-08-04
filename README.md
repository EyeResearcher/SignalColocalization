# Signal Colocalization

This repository segments a reference channel with Cellpose, measures one or more
signal channels inside each mask, and reports per-cell and per-image
colocalization statistics. It supports an interactive notebook workflow and a
JSON-configured command-line workflow.

## What you need

Required inputs:

- A directory containing microscopy images. Supported extensions are `.oir`,
  `.czi`, `.tif`, `.tiff`, `.ome.tif`, and `.ome.tiff`.
- At least one **reference channel** to segment. Select it by zero-based channel
  number or, when present in the image metadata, by channel name.
- A Cellpose model for each reference set. `model` can be a Cellpose model-zoo
  name or the path to a custom trained model. A custom model file must be
  available locally. A model-zoo model may be downloaded by Cellpose on first
  use, so that first run may require internet access.
- At least one **signal channel** to measure within the reference masks.
- An output directory. It is created automatically if it does not exist.

Images are read as channel/Z/Y/X (`CZYX`) stacks. The selected time point and
scene are read, then the Z axis is reduced to a two-dimensional image before
segmentation. All configured reference and signal channels must exist in every
input image.

## Setup

Clone or download the repository, open a terminal in its root directory, and
create a virtual environment. Python 3.10 or newer is recommended.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include Cellpose, PyTorch, microscopy readers, Jupyter,
scientific Python packages, and Matplotlib. With `device="auto"`, the pipeline
uses CUDA when available, Apple MPS when available, and otherwise the CPU. A
working GPU-specific PyTorch installation may need to be installed separately
for your CUDA environment; follow the PyTorch installation instructions for the
machine before running the analysis.

Confirm that the command-line interface is available:

```powershell
python -m colocalize --help
```

## Configuration and accepted inputs

Channel numbers are zero-based. For example, `channel: 3` selects the fourth
channel. A metadata name such as `channel: "DAPI"` can be used instead when that
name is present in every image.

Reference-set options:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | Unique label used in tables and filenames |
| `channel` | yes | Zero-based index or metadata channel name |
| `model` | yes in practice | Cellpose model-zoo name or custom model path; the code default is `cpdino_BRN3A` |
| `diameter` | no | Expected object diameter in pixels; `null` lets Cellpose choose |
| `flow_threshold` | no | Cellpose flow-error threshold; default `0.4` |
| `cellprob_threshold` | no | Cellpose cell-probability threshold; default `0.0` |
| `min_size` | no | Minimum object size in pixels; default `15` |
| `normalize` | no | Ask Cellpose to normalize the image; default `true` |

Signal-channel options:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | Unique signal label used in output column names |
| `channel` | yes | Zero-based index or metadata channel name |
| `threshold_method` | no | `otsu`, `percentile`, `absolute`, or `none`; default `otsu` |
| `threshold_value` | sometimes | Percentile (default `99`) or absolute intensity cutoff; required for `absolute` |
| `positive_fraction_cutoff` | no | Fraction of mask pixels above threshold needed to call a cell positive; default `0.80` |

Important analysis options include:

- `z_projection`: `max`, `mean`, or `first` (default `max`).
- `time_index` and `scene_index`: zero-based indices (both default to `0`).
- `recursive`: search subdirectories when `true` (default `false`).
- `exclude`: filenames, relative paths, absolute paths, or glob patterns to skip.
- `device`: `auto`, `cpu`, `mps`, `cuda`, or a device such as `cuda:1`.
- `save_masks`: save label TIFFs under `OUTPUT_DIR/masks` (default `true`).
- `extensions`: optional list overriding the supported extension list.

The output directory is automatically excluded from image discovery, so an
output directory nested beneath the input directory will not be reprocessed.

## Run the notebook

Start Jupyter and open the included notebook:

```powershell
jupyter lab run_colocalization.ipynb
```

Then run the notebook in order:

1. Run the import cell.
2. Edit the `AnalysisConfig` cell. Set `input_dir`, `output_dir`, reference
   channel/model settings, and signal channel/threshold settings.
3. Run `inspect_inputs(config)`. Check the discovered files, `shape_cyx`, and
   channel names before starting Cellpose.
4. Run `result = run_analysis(config)`.
5. Run the visualization cell. Set `img_number` to the row to inspect; the cell
   loads the corresponding saved mask and calls `show_segmentation(...)`.

The notebook uses a configurable transform list. `MaxProjection()`,
`MeanProjection()`, or `SelectPlane(index)` can reduce Z, and transforms are
applied in the order listed. This is the preferred interface if you need custom
Python transforms beyond the command-line `z_projection` choices.

## Run from the command line

Copy [analysis_config.example.json](analysis_config.example.json), rename it,
and edit its paths, channels, model, and thresholds. Relative `input_dir` and
`output_dir` values are resolved relative to the JSON file. Use forward slashes,
escaped backslashes, or absolute paths in JSON.

Inspect inputs without running Cellpose:

```powershell
python -m colocalize analysis_config.json --inspect
```

Run the analysis:

```powershell
python -m colocalize analysis_config.json
```

Open each segmentation visualization after the analysis. Close the current
Matplotlib window to advance to the next image/reference/signal combination:

```powershell
python -m colocalize analysis_config.json --show-segmentation
```

For a non-interactive or remote run, save the same visualizations as PNG files:

```powershell
python -m colocalize analysis_config.json --save-segmentation
```

Both flags can be combined. Segmentation plots are saved under
`OUTPUT_DIR/segmentation_qc`. When either visualization flag is used, masks are
saved even if the JSON sets `save_masks` to `false`, because the QC plots require
them.

## Outputs

Each run writes:

- `cells.csv`: one row per segmented mask, including morphology, reference
  intensity, signal intensity, positive area/fraction, positive-cell calls,
  Pearson correlation, Jaccard overlap, and Manders coefficients.
- `image_summary.csv`: cell counts and aggregate measurements for each input
  image/reference-set pair.
- `masks/*.tif`: integer Cellpose label images when `save_masks` is enabled.
- `segmentation_qc/*.png`: optional four-panel QC figures created with
  `--save-segmentation`.

The QC figure shows the scaled reference image, scaled signal image, red mask
boundaries, and a false-color overlay. In the overlay the reference is green,
the signal is magenta, all masks are yellow, and signal-positive/colocalized
masks are cyan.

## Troubleshooting

- **No images found:** check `input_dir`, `recursive`, `exclude`, and file
  extensions. Run with `--inspect` first.
- **Channel not found/out of range:** use the `--inspect` table to confirm
  zero-based indices and metadata names for every acquisition.
- **Custom model cannot be loaded:** use the full path to the trained Cellpose
  model and confirm that the environment can read it.
- **Interactive plots do not appear:** use `--save-segmentation` on headless
  systems. On a desktop, make sure Matplotlib has an interactive GUI backend.
- **Out of memory:** process fewer images at a time, use CPU, or reduce image
  dimensions before analysis.
