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
- A Cellpose model for each reference set. The included `cpdino_BRN3A` and
  `cpdino_RPBMS` models download automatically from Hugging Face on first use.
  `model` can also be a Cellpose model-zoo name, local custom-model path, or
  explicit Hugging Face model reference. The first run may require internet
  access.
- At least one **signal channel** to measure within the reference masks.
- An output directory. It is created automatically if it does not exist.

Images are read as channel/Z/Y/X (`CZYX`) stacks. The selected time point and
scene are read, then the Z axis is reduced to a two-dimensional image before
segmentation. All configured reference and signal channels must exist in every
input image.

## Setup

Python 3.12 or newer is required. The recommended setup uses a dedicated
Conda environment so the analysis dependencies remain isolated from the rest of
your Python installation.

### Conda (recommended)

From the directory where you keep projects, create the environment and clone
the repository **once**. The `conda run` commands force installation into the
named environment even if shell activation is not configured correctly:

```powershell
conda create --name signal-colocalization python=3.12 -y
git clone https://github.com/EyeResearcher/SignalColocalization.git
cd SignalColocalization
conda run --name signal-colocalization python -m pip install --upgrade pip
conda run --name signal-colocalization python -m pip install --editable .
conda run --name signal-colocalization python -m colocalize --help
```

If the `SignalColocalization` directory already exists, do not run `git clone`
again from inside it. Enter the existing repository directory and start with
the `conda run` installation commands instead.

In Google Colab, install the cloned repository and all of its dependencies into
the active runtime with:

```python
%pip install -q -e /content/SignalColocalization/
```

The editable install means changes pulled into the repository are immediately
used without reinstalling the `colocalize` package. Rerun the installation only
when `requirements.txt` or `pyproject.toml` changes.

Activate the environment before running notebooks or commands interactively:

```powershell
conda activate signal-colocalization
cd path/to/SignalColocalization
python --version
```

The terminal prompt should begin with `(signal-colocalization)`, and
`python --version` should report Python 3.12 or newer. Do not install the
requirements when the prompt still begins with `(base)`, because that can
conflict with packages installed by Anaconda, Spyder, or other projects.

If you previously created this environment with Python 3.11, upgrade it before
installing the requirements:

```powershell
conda activate signal-colocalization
conda install python=3.12 -y
python --version
cd path/to/SignalColocalization
conda run --name signal-colocalization python -m pip install --editable .
```

### Python virtual environment (alternative)

If Conda is not installed, first clone the repository and enter it:

```powershell
git clone https://github.com/EyeResearcher/SignalColocalization.git
cd SignalColocalization
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --editable .
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable .
```

The requirements include Cellpose, PyTorch, microscopy readers, Jupyter,
the official DINOv3 implementation required by CPDINO models, scientific Python
packages, and Matplotlib. Because DINOv3 is installed from its GitHub repository,
Git must be available during installation. With `device="auto"`, the pipeline
uses CUDA when available, Apple MPS when available, and otherwise the CPU. A
working GPU-specific PyTorch installation may need to be installed separately
for your CUDA environment; follow the PyTorch installation instructions for the
machine before running the analysis.

If the requirements were installed before DINOv3 was added and a CPDINO model
fails with `NameError: name 'dinov3_vitl16' is not defined`, install the missing
dependency into the project environment and verify its import:

```powershell
conda run --name signal-colocalization python -m pip install "git+https://github.com/facebookresearch/dinov3.git@6876159a11b4df116f30f667f8c9888617df0751"
conda run --name signal-colocalization python -c "from dinov3.hub.backbones import dinov3_vitl16; print('DINOv3 ready')"
```

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
| `model` | yes in practice | Bundled retinal-model name, `hf://` reference, Cellpose model-zoo name, or local path; default `cpdino_BRN3A` |
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
- `save_segmentation`: save the QC grid and all four individual panels for
  every processed image/reference/signal combination (default `false`).
- `segmentation_output_dir`: optional destination for QC PNGs. Setting it also
  enables `save_segmentation`; the default destination is
  `OUTPUT_DIR/segmentation_qc`.
- `extensions`: optional list overriding the supported extension list.

The output directory is automatically excluded from image discovery, so an
output directory nested beneath the input directory will not be reprocessed.

### Retinal model downloads

The fine-tuned weights are hosted in the public
[mmzinn12/cellpose-retinal-models](https://huggingface.co/mmzinn12/cellpose-retinal-models)
repository. These configuration values are recognized and downloaded
automatically:

- `cpdino_BRN3A`
- `cpdino_RPBMS`

Hugging Face stores the downloaded file in its local cache, so subsequent runs
reuse it. If a file with the same name already exists in
`~/.cellpose/models`, that local Cellpose copy is preferred. Each model is
approximately 1.16 GB.

An arbitrary file in another Hugging Face model repository can be selected with
this syntax:

```json
"model": "hf://OWNER/REPOSITORY/PATH/TO/MODEL"
```

Local model paths and Cellpose built-in names such as `cpdino` and `cpsam_v2`
continue to work normally.

## Run the notebook

Start Jupyter and open the included notebook:

```powershell
jupyter lab run_colocalization.ipynb
```

Then run the notebook in order:

1. Run the import cell.
2. Edit the `AnalysisConfig` cell. Set `input_dir`, `output_dir`, reference
   channel/model settings, and signal channel/threshold settings. Set
   `save_segmentation=True` to export QC PNGs for every input image.
3. Run `inspect_inputs(config)`. Check the discovered files, `shape_cyx`, and
   channel names before starting Cellpose.
4. Run `result = run_analysis(config)`.
5. Run the visualization cell. Set `img_number` to the row to inspect; the cell
   loads the corresponding saved mask and calls `show_segmentation(...)`.

From a notebook, the matching save helper accepts the reference image, masks,
signal image, and an explicit destination:

```python
visualization.save_segmentation_views(
    reference_image,
    masks,
    signal_image,
    signal_spec=config.signal_channels[0],
    output_dir=Path("path/to/segmentation-qc"),
    name="example_RPBMS_GD",
    title="Example acquisition",
)
```

It returns paths to the saved grid, reference, signal, masks, and overlay PNGs.

The notebook uses a configurable transform list. `MaxProjection()`,
`MeanProjection()`, or `SelectPlane(index)` can reduce Z, and transforms are
applied in the order listed. This is the preferred interface if you need custom
Python transforms beyond the command-line `z_projection` choices.

## Run from the command line

The JSON configuration is optional. A single-reference, single-signal analysis
can be configured entirely with command-line arguments:

```powershell
python -m colocalize `
  --input-dir "path/to/images" `
  --output-dir "path/to/results" `
  --reference-name RPBMS `
  --reference-channel 3 `
  --model cpdino_RPBMS `
  --signal-name GD `
  --signal-channel 2 `
  --threshold-method percentile `
  --threshold-value 99.5 `
  --positive-fraction-cutoff 0.05
```

Every serializable analysis setting has a command-line option. Run
`python -m colocalize --help` for the complete list. When omitted, the primary
defaults are input directory `.`, output directory `results`, reference channel
`0`, model `cpdino_BRN3A`, signal channel `1`, Otsu thresholding, max Z
projection, automatic device selection, and saved masks.

Inspect inputs without running Cellpose by adding `--inspect`, or download the
selected model and exit by adding `--download-models`.

For multiple reference sets or signal channels, copy
[analysis_config.example.json](analysis_config.example.json), rename it, and
edit its values. Relative `input_dir` and `output_dir` values are resolved
relative to the JSON file. Explicit command-line options override JSON values;
reference or signal options override the first corresponding entry.

Inspect inputs without running Cellpose:

```powershell
python -m colocalize analysis_config.json --inspect
```

Optionally download all configured Hugging Face models before starting a long
analysis:

```powershell
python -m colocalize analysis_config.json --download-models
```

This prints the resolved local path for each reference model and exits. Model
downloads also happen automatically during a normal analysis, so this step is
not required.

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

Each image/reference/signal combination produces five files: the complete
four-panel grid plus separate `reference`, `signal`, `masks`, and `overlay`
PNGs. To save them to a specific folder, provide the folder directly; this
option implies `--save-segmentation`:

```powershell
python -m colocalize analysis_config.json `
  --segmentation-output-dir "C:\path\to\segmentation-qc"
```

The display and save flags can be combined. Without an explicit folder,
segmentation plots are saved under `OUTPUT_DIR/segmentation_qc`. When either
the config or CLI enables saving, all images discovered from `input_dir` are
exported. Interactive display still requires saved masks when it is requested
after analysis.

## Run from Johnson Lab eLabFTW or other image pointers

`run_elab_colocalization.py` is the read-only integration entry point for the
lab website/eLabFTW deployment. It resolves individual images or groups of
images into a temporary staging directory, runs the same configured pipeline,
and deletes staged copies when the process exits. It never writes data back to
eLabFTW.

Connect to the JHU VPN and set the API key in the environment. Do not add the
key to JSON, source code, notebooks, or Git:

```powershell
$env:ELAB_APIKEY = "your-api-key"
$env:ELAB_BASE = "https://johnsonlab.wilmer.jhu.edu/api/v2" # optional; this is the default
python run_elab_colocalization.py --smoke-test
```

The smoke test performs a single extended experiment list request and exits.
For analysis, repeat `--source POINTER` as needed. Accepted pointers are:

| Pointer | Resolves to |
| --- | --- |
| `experiment:42` | All supported image uploads on experiment 42 |
| `item:9` or `resource:9` | All supported image uploads on resource 9 |
| `upload:123` | One eLabFTW upload, independent of its parent record |
| An eLabFTW experiment/resource/upload URL | The record group or individual upload named by the URL |
| `file:path/to/image.oir` or a bare file path | One local microscopy image |
| `dir:path/to/images` or a bare directory | Supported images below a local directory, recursively |
| `glob:path/**/*.ome.tif` | Images matching a local glob |
| `@sources.txt` | A UTF-8 manifest with one pointer per line; blank lines and `#` comments are allowed |

Manifest-relative file, directory, glob, and nested-manifest paths resolve from
the manifest's directory. Repeated or overlapping pointers are deduplicated.
`--experiment-id`, `--item-id`, and `--upload-id` are repeatable shorthands for
their corresponding `--source` forms.

Run a mixed group using the same analysis JSON as the local CLI; its
`input_dir` is ignored and its `output_dir` can be overridden:

```powershell
python run_elab_colocalization.py analysis_config.json `
  --source experiment:42 `
  --source resource:9 `
  --source "glob:C:/microscopy/validation/*.oir" `
  --output-dir results/combined `
  --save-segmentation
```

The eLabFTW runner also accepts `--segmentation-output-dir PATH`; it saves the
four-panel grid and all four individual panel PNGs to that directory.

Use `--inspect` to resolve and inspect image shapes/channels without running
Cellpose. A single shorthand `--experiment-id 42` retains the filename
`experiment_42_colocalization.xlsx`; mixed-source runs produce
`colocalization_results.xlsx`. The workbook contains record-level custom fields,
source-image provenance, image summaries, and per-cell measurements. Every
analysis row carries its original pointer, upload ID and parent record when
available. Existing CSV, mask, and optional QC outputs are retained.

Tom's deployment needs the complete repository plus the packages in
`requirements.txt`. The integration-specific addition is `openpyxl>=3.1` for
Excel output. The server should inject `ELAB_APIKEY`; `ELAB_BASE` is optional.

### eLab-facing configuration form

[web/colocalization_configurator.html](web/colocalization_configurator.html) is
a self-contained interface Tom can place in the lab website or adapt to its
house style. It exposes repeatable image sources, every reference-set and
signal-channel setting, acquisition selection, projection, compute device,
file discovery, masks, QC, inspection mode, and output location. It never asks
for an API key.

The form downloads a versioned `colocalization_job.json`. Run that exact job
without translating it into a second configuration format:

```powershell
python run_elab_colocalization.py --job colocalization_job.json
```

The formal interface contract is
[colocalize/colocalization_job.schema.json](colocalize/colocalization_job.schema.json).
Website code can use it to generate or validate another UI, and the deployed
runner can expose it directly:

```powershell
python run_elab_colocalization.py --print-schema
```

Legacy `analysis_config.json` plus command-line source arguments remains
supported. A job file and legacy config cannot be supplied together.

## Outputs

Each run writes:

- `cells.csv`: one row per segmented mask, including morphology, reference
  intensity, signal intensity, positive area/fraction, positive-cell calls,
  Pearson correlation, Jaccard overlap, and Manders coefficients.
- `image_summary.csv`: cell counts and aggregate measurements for each input
  image/reference-set pair.
- `masks/*.tif`: integer Cellpose label images when `save_masks` is enabled.
- `segmentation_qc/*.png`: optional four-panel QC grids and their four separate
  panel images, created with `--save-segmentation`.

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
  model and confirm that the environment can read it. For a Hugging Face model,
  confirm internet access and that `huggingface_hub` is installed.
- **Interactive plots do not appear:** use `--save-segmentation` on headless
  systems. On a desktop, make sure Matplotlib has an interactive GUI backend.
- **Out of memory:** process fewer images at a time, use CPU, or reduce image
  dimensions before analysis.
