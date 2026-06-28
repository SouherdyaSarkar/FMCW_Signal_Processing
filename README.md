# Radar Data Processor

Jupyter notebooks and Python scripts for processing mmWave radar ADC data using the [openradar](https://github.com/PreSenseRadar/OpenRadar) open-source library.

## Directory Structure & Processors Overview

This directory contains several processors for mmWave radar processing (including FFTs, Capon beamforming, and phase tracking) as well as IMU vibration logging.

### 1. Main Pipeline Processors (`/processors`)

*   **[pipeline_cleaned.ipynb](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/pipeline_cleaned.ipynb)**: The main, cleaned radar signal processing pipeline. Includes:
    *   **FFT Analysis**: Range and Doppler FFT processing with dynamic GIF visualization.
    *   **Phase Analysis**: Phase extraction and cleaning across 182 and 546 chirps, including bandpass filtering.
    *   **Acceleration Computation**: Calculation and plotting of target vibration acceleration from the filtered phase history.
    *   **Comparison Plotting**: Phase and acceleration comparison across multiple datasets.
*   **[Full_pipeline.ipynb](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/Full_pipeline.ipynb)**: The comprehensive development pipeline, covering range/Doppler FFT, phase difference calculation (including wrapped/unwrapped comparison, Kalman filtering, and GIF animation of phase histories), and target acceleration mapping based on a revised paper formula.
*   **[Data_processor.ipynb](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/Data_processor.ipynb)**: Standard radar processing workflow implementing Range FFT (finding peak frequencies/bins), Doppler FFT (Range-Doppler estimation), and Azimuth FFT (Angle of Arrival estimation).

### 2. Specialized Folders

#### 📂 [Acceleration](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/Acceleration/)
*   **[Acceleration_calc.ipynb](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/Acceleration/Acceleration_calc.ipynb)**: Extract target phase, perform phase unwrapping, and compute/plot vibration acceleration over time from downsampled radar data.

#### 📂 [MVDR_Phase (Capon Beamforming)](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/MVDR_Phase/)
*   **[Capon_beamforming.ipynb](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/MVDR_Phase/Capon_beamforming.ipynb)**: Implements MVDR (Minimum Variance Distortionless Response) Capon Beamforming to resolve AoA, detect target range bins, extract phase information, and perform reference-based noise/jitter cancellation.
*   **[Capon_beamforming.py](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/MVDR_Phase/Capon_beamforming.py)**: The Python script version of the Capon Beamforming processor.

#### 📂 [IMU Processors](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/IMU%20Processors/)
*   **[IMUProccessor.ipynb](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/IMU%20Processors/IMUProccessor.ipynb)**: Processes IMU acceleration CSV logs for different depths (5cm, 10cm, 15cm) and motor power levels (0% to 100%), generating stacked comparison plots of acceleration components.
*   **[Acceleration_plot.ipynb](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/IMU%20Processors/Acceleration_plot.ipynb)**: Processes motor vibration CSV logs from multiple devices, splits records by device ID, and plots rolling RMS values of acceleration components ($x, y, z$) and magnitude over time.
*   **[IMUProcessor.py](file:///c:/Users/Souherdya/OneDrive/Desktop/Signal%20Processing%20IIT%20KGP/Range_Doppler_AOA_code/processors/IMU%20Processors/IMUProcessor.py)**: Implements orientation integration (roll, pitch, yaw) and global acceleration correction (removing gravity component) to integrate IMU velocity/trajectory into 3D space plots.


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
