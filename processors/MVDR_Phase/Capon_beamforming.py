import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../..')))

from openradar.mmwave.dataloader import DCA1000
from openradar.mmwave import dsp
from openradar.mmwave.dsp.utils import Window

# Configuration & File Paths
DATA_PATH = "data/adc_data_2026-05-26_15-17-41_phone_vib.npy"
numFrames = 200
numADCSamples = 256
numTxAntennas = 3
numRxAntennas = 4
numLoopsPerFrame = 182
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame


# Object Angle in degrees
OBJ_ANGLE = 12.0 

def gen_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    """Vectorized Steering Vector Generation"""
    num_vec = int(round(2 * ang_est_range / ang_est_resolution + 1))
    angles_rad = np.radians(np.linspace(-ang_est_range, ang_est_range, num_vec))
    ant_indices = np.arange(num_ant)
    
    mag = -np.pi * np.outer(ant_indices, np.sin(angles_rad))
    steering_vectors = np.exp(1j * mag).T
    
    return num_vec, steering_vectors

def cov_matrix(x):
    if x.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")
    if x.shape[0] > x.shape[1]:
        warnings.warn("cov_matrix input should have Vrx as rows. Needs to be transposed", RuntimeWarning)
        x = x.T
    _, num_adc_samples = x.shape
    Rxx = x @ np.conjugate(x.T)
    Rxx = np.divide(Rxx, num_adc_samples)
    return Rxx

def forward_backward_avg(Rxx):
    assert np.size(Rxx, 0) == np.size(Rxx, 1)
    M = np.size(Rxx, 0)  
    Rxx = np.matrix(Rxx)  
    J = np.fliplr(np.eye(M))  
    J = np.matrix(J)  
    R_fb = 0.5 * (Rxx + J * np.conjugate(Rxx) * J)
    return np.array(R_fb)

def aoa_capon(x, steering_vector, magnitude=False):
    if steering_vector.shape[1] != x.shape[0]:
        raise ValueError("'steering_vector' shape mismatch.")

    Rxx = cov_matrix(x)
    Rxx = forward_backward_avg(Rxx)
    
    diagonal_loading = 1e-3 * np.trace(Rxx) * np.eye(x.shape[0])
    Rxx_loaded = Rxx + diagonal_loading
    Rxx_inv = np.linalg.inv(Rxx_loaded)
    
    first = Rxx_inv @ steering_vector.T
    den = np.reciprocal(np.einsum('ij,ij->i', steering_vector.conj(), first.T))
    weights = first * den
    
    if magnitude:
        return np.abs(den), weights
    else:
        return den, weights

def iterative_range_bins_detection(rangeResult, min_bin=10, max_bin=None):
    rangeResult = np.transpose(np.stack([rangeResult[0::3], rangeResult[1::3], rangeResult[2::3]], axis=1), axes=(1,2,0,3))
    range_result_absnormal_split = []
    
    for i in range(numTxAntennas):
        for j in range(numRxAntennas):
            r_r = np.abs(rangeResult[i][j])
            r_r[:, :min_bin] = 0 
            if max_bin is not None:
                r_r[:, max_bin:] = 0 
                
            min_val, max_val = np.min(r_r), np.max(r_r)
            r_r_normalise = np.zeros_like(r_r) if max_val == min_val else (r_r - min_val) / (max_val - min_val) * 1000
            range_result_absnormal_split.append(r_r_normalise)
    
    range_abs_combined_nparray = np.sum(range_result_absnormal_split, axis=0) / (numTxAntennas * numRxAntennas)
    range_abs_combined_nparray_collapsed = np.sum(range_abs_combined_nparray, axis=0) / numLoopsPerFrame
    peaks_min_intensity_threshold = np.argsort(range_abs_combined_nparray_collapsed)[::-1][:max_bin-min_bin]
    max_range_index = np.argmax(range_abs_combined_nparray_collapsed)
    
    return max_range_index, peaks_min_intensity_threshold, rangeResult


frame_index = 2

if __name__ == "__main__":
    save_dir = "Simulations/Capon_Beamformed_Phase_setzero"
    os.makedirs(save_dir, exist_ok=True)

    loaded_adc_data = np.load(DATA_PATH)
    current_frame = loaded_adc_data[frame_index-1:frame_index]
    
    adc_data = np.apply_along_axis(DCA1000.organize, 1, current_frame,num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)
    radar_cube = dsp.range_processing(adc_data[0], window_type_1d=Window.BLACKMAN)
    
    min_b = 10
    max_b = 20
    _, selected_bins, rangeResult = iterative_range_bins_detection(radar_cube, min_bin=min_b, max_bin=max_b)
    bins_to_process = selected_bins[:]  
    print(f"Top 10 sorted bins targeted: {bins_to_process}")
    
    scan_range_deg = 60
    resolution_deg = 1
    num_virtual_antennas = numTxAntennas * numRxAntennas 
    
    num_vec, steering_vectors = gen_steering_vec(scan_range_deg, resolution_deg, num_virtual_antennas)
    angles_grid = np.linspace(-scan_range_deg, scan_range_deg, num_vec)
    
    for bin_no in bins_to_process:
        x_slice = rangeResult[:, :, :, bin_no].reshape(num_virtual_antennas, numLoopsPerFrame)
        
        capon_spectrum, capon_weights_matrix = aoa_capon(x_slice, steering_vectors, magnitude=True)
        capon_spectrum_dB = 10 * np.log10(capon_spectrum / np.max(capon_spectrum))
        
        target_angle_idx = np.argmin(np.abs(angles_grid - OBJ_ANGLE))
        actual_steered_angle = angles_grid[target_angle_idx]
        
        w_opt = capon_weights_matrix[:, target_angle_idx]
        
        filtered_signal = np.dot(w_opt.conj().T, x_slice)
        
        capon_filtered_phase = np.unwrap(np.angle(filtered_signal))
        
        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot A: The Spatial Spectrum (Intermediate Showcase)
        ax1.plot(angles_grid, capon_spectrum_dB, color='teal', linewidth=2)
        ax1.axvline(actual_steered_angle, color='red', linestyle='--', alpha=0.7, 
                    label=f'Steered Direction (OBJ_ANGLE: {actual_steered_angle:.1f}°)')
        
        # Let's also mark where the algorithm *thinks* the absolute peak is
        max_peak_angle = angles_grid[np.argmax(capon_spectrum)]
        ax1.axvline(max_peak_angle, color='orange', linestyle=':', alpha=0.7, 
                    label=f'Max Power Peak at {max_peak_angle:.1f}°')
                    
        ax1.set_title(f"Range Bin: {bin_no} | Capon Spatial Spectrum")
        ax1.set_xlabel("Angle (Degrees)")
        ax1.set_ylabel("Power Density (dB)")
        ax1.set_ylim(-35, 5)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()
        
        # Plot B: The Extracted Phase 
        ax2.plot(capon_filtered_phase, color='darkmagenta', marker='.', markersize=4)
        ax2.set_title(f"Noise-Filtered Absolute Phase (Steered at {actual_steered_angle:.1f}°)")
        ax2.set_xlabel("Chirp Index")
        ax2.set_ylabel("Phase (Radians)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"frame{frame_index}_bin{bin_no}_capon_combined.png"))
        plt.close(fig)
        
    print(f"\n[SUCCESS] Custom Capon phase processing complete. Visual summaries saved to:\n{os.path.abspath(save_dir)}")