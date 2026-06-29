# Radar Data Processor

Jupyter notebooks for processing mmWave radar ADC data using the [openradar](https://github.com/PreSenseRadar/OpenRadar) open-source library.

## Setup

### 1. Clone openradar

The `openradar` package must be cloned into the **parent directory** of this folder (i.e., one level above `processors/`):

```bash
git clone https://github.com/PreSenseRadar/OpenRadar.git openradar
```

Your directory structure should look like:

```
project-root/
├── openradar/       ← cloned repo
├── data/            ← restricted data folder (see below)
└── processors/      ← this repo
```

### 2. Install openradar

```bash
cd openradar
pip install -e .
```

### 3. Create a virtual environment

From the `processors/` directory (or project root):

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# or
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### 4. Register the kernel (for Jupyter)

```bash
python -m ipykernel install --user --name=.venv
```

## Data Access

The notebooks reference a `../data/` folder via the `DATA_PATH` variable, for example:

```python
DATA_PATH = "../data/adc_data_2026-02-06_16-35-07_land_vib_80.npy"
```

This folder contains raw ADC capture files and is **not included in this repository**. Access is restricted to authorized personnel only. Contact the project maintainer to request access.

## Usage

Open any notebook in `processors/` with Jupyter and select the `.venv` kernel. Make sure the `data/` folder is in place before running data loading cells.
