import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from openradar.mmwave.dataloader import DCA1000
from openradar.mmwave import dsp
from openradar.mmwave.dsp.utils import Window
import time
from datetime import datetime
import math


DATA_PATH_VIB = "data/Mobile_data/adc_data_2026-06-03_16-05-58_pho_vib.npy" 
DATA_PATH_STATIC = "data/Mobile_data/adc_data_2026-06-03_15-52-17_pho_static.npy"

numFrames = 600
numADCSamples = 256
numTxAntennas = 3
numRxAntennas = 4
numLoopsPerFrame = 182
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

stop_flag = False
start_flag = True  
Tp = 14e-6
Tc = 72e-6

def get_pcd(det_matrix):
    fft2d_sum = det_matrix.astype(np.int64)
    thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                axis=0,
                                                                arr=fft2d_sum.T,
                                                                l_bound=1.5,
                                                                guard_len=4,
                                                                noise_len=16)

    thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                            axis=0,
                                                            arr=fft2d_sum,
                                                            l_bound=2.5,
                                                            guard_len=4,
                                                            noise_len=16)

    thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
    det_doppler_mask = (det_matrix > thresholdDoppler)
    det_range_mask = (det_matrix > thresholdRange)

    full_mask = (det_doppler_mask & det_range_mask)
    det_peaks_indices = np.argwhere(full_mask == True)

    peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

    dtype_location = '(' + str(numTxAntennas) + ',)<f4'
    dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
    detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
    detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
    detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
    detObj2DRaw['peakVal'] = peakVals.flatten()
    detObj2DRaw['SNR'] = snr.flatten()

    detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)
    detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)
    SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
    peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
    detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5, range_resolution)

    azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]

    Psi, Theta, Ranges, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'],
                                                                    range_resolution, method='Bartlett')
    return xyzVec

def iterative_range_bins_detection(rangeResult, min_bin=10, max_bin=None):
    rangeResult = np.transpose(np.stack([rangeResult[0::3], rangeResult[1::3], rangeResult[2::3]], axis=1),axes=(1,2,0,3))
    range_result_absnormal_split=[]
    
    for i in range(numTxAntennas):
        for j in range(numRxAntennas):
            r_r=np.abs(rangeResult[i][j])
            
            r_r[:, :min_bin] = 0
            if max_bin is not None:
                r_r[:, max_bin:] = 0
                
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            
            if max_val == min_val: 
                r_r_normalise = np.zeros_like(r_r)
            else:
                r_r_normalise = (r_r - min_val) / (max_val - min_val) * (1000 - 0) + 0
                
            range_result_absnormal_split.append(r_r_normalise)
    
    range_abs_combined_nparray=np.zeros((numLoopsPerFrame,numADCSamples))
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray+=ele
    range_abs_combined_nparray/=(numTxAntennas*numRxAntennas)
    
    range_abs_combined_nparray_collapsed=np.sum(range_abs_combined_nparray,axis=0)/numLoopsPerFrame
    peaks_min_intensity_threshold = np.argsort(range_abs_combined_nparray_collapsed)[::-1][:max_bin-min_bin]
    max_range_index=np.argmax(range_abs_combined_nparray_collapsed)
    
    return max_range_index, peaks_min_intensity_threshold, rangeResult

def get_phase(r,i):
    if r==0:
        if i>0:
            phase=np.pi/2
        else :
            phase=3*np.pi/2
    elif r>0:
        if i>=0:
            phase=np.arctan(i/r)
        if i<0:
            phase=2*np.pi - np.arctan(-i/r)
    elif r<0:
        if i>=0:
            phase=np.pi - np.arctan(-i/r)
        else:
            phase=np.pi + np.arctan(i/r)
    return phase

def phase_unwrapping(phase_len,phase_cur_frame):
    i=1
    new_signal_phase = phase_cur_frame
    for k,ele in enumerate(new_signal_phase):
        if k==len(new_signal_phase)-1:
            continue
        if new_signal_phase[k+1] - new_signal_phase[k] > 1.5*np.pi:
            new_signal_phase[k+1:] = new_signal_phase[k+1:] - 2*np.pi*np.ones(len(new_signal_phase[k+1:]))
    return np.array(new_signal_phase)

def get_phase_antennawise(range_FFT_,peak):
    phase_per_antenna=[]
    for k in range(0,numLoopsPerFrame):
        r = range_FFT_[k][peak].real
        i = range_FFT_[k][peak].imag
        phase=get_phase(r,i)
        phase_per_antenna.append(phase)
    phase_cur_frame=phase_unwrapping(len(phase_per_antenna),phase_per_antenna)
    return phase_cur_frame

def get_averaged_phase(rangeResult, target_bin):
    all_phases = []
    for tx in range(numTxAntennas):
        for rx in range(numRxAntennas):
            unwrapped_phase = get_phase_antennawise(rangeResult[tx][rx], target_bin)
            all_phases.append(unwrapped_phase)
    all_phases = np.array(all_phases)
    avg_phase = np.mean(all_phases, axis=0)
    return avg_phase

def extract_phase_diffs_from_file(data_path, start_frame, num_frames, target_bins):
    print(f"Loading data from: {data_path}...")
    loaded_adc_data = np.load(data_path)
    accumulated_phase_diffs = {bin_idx: [] for bin_idx in target_bins}
    
    for f in range(start_frame, start_frame + num_frames + 1):  
        current_frame = loaded_adc_data[f-1 : f]
        adc_data = np.apply_along_axis(DCA1000.organize, 1, current_frame, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)
        radar_cube = dsp.range_processing(adc_data[0], window_type_1d=Window.BLACKMAN)
        
        min_b = 0
        max_b = 20
        _, _, rangeResult = iterative_range_bins_detection(radar_cube, min_bin=min_b, max_bin=max_b)

        for target_bin in target_bins:
            avg_phase = get_averaged_phase(rangeResult, target_bin)
            phase_diff = np.insert(np.diff(avg_phase), 0, 0)
            accumulated_phase_diffs[target_bin].extend(phase_diff)
            
    return accumulated_phase_diffs

def plot_cross_recording_comparison(vib_data, static_data, subtracted_data, target_bin, start_frame, num_frames, save_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f"Cross-Recording Phase Subtraction Trial - Range Bin: {target_bin}\n(Frames {start_frame} to {start_frame + num_frames})", fontsize=14, fontweight='bold')
    
    axes[0].plot(vib_data, color='dodgerblue', markersize=0.5, linestyle='-', linewidth=0.5, alpha=0.9)
    axes[0].set_title("Vibration Data", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Phase Diff (Rad)")
    axes[0].grid(True, linestyle=':', alpha=0.7)
    
    # 2. Static
    axes[1].plot(static_data, color='darkorange', markersize=0.5, linestyle='-', linewidth=0.5, alpha=0.9)
    axes[1].set_title("Static Baseline Data", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Phase Diff (Rad)")
    axes[1].grid(True, linestyle=':', alpha=0.7)

    # 3. Subtracted Result
    axes[2].plot(subtracted_data, color='red', markersize=0.5, linestyle='-', linewidth=0.5, alpha=0.9)
    axes[2].set_title("Result: (Vibration - Static)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Cumulative Chirp Index (Time)")
    axes[2].set_ylabel("Phase Diff (Rad)")
    axes[2].grid(True, linestyle=':', alpha=0.7)
    
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"frames{start_frame}_to_{start_frame+num_frames}_bin{target_bin}_cross_subtracted.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved cross-recording comparison plot for bin {target_bin} at {save_path}")

if __name__ == "__main__":
    start_frame = 300
    num_frames = 100  
    target_bins = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    save_directory = "Simulations/Radar/Phone_data_day02/Cross_Subtraction_Trial"
    
    print("--- STARTING CROSS-RECORDING SUBTRACTION TRIAL ---")
    
    print("\n[STEP 1] Extracting Vibration Phase Data...")
    vib_phase_diffs = extract_phase_diffs_from_file(DATA_PATH_VIB, start_frame, num_frames, target_bins)
    
    print("\n[STEP 2] Extracting Static Baseline Phase Data...")
    static_phase_diffs = extract_phase_diffs_from_file(DATA_PATH_STATIC, start_frame, num_frames, target_bins)
    
    print("\n[STEP 3] Performing Subtraction and Plotting...")
    for target_bin in target_bins:
        vib_array = np.array(vib_phase_diffs[target_bin])
        static_array = np.array(static_phase_diffs[target_bin])
        
        subtracted_array = vib_array - static_array

        plot_cross_recording_comparison(
            vib_data=vib_array, 
            static_data=static_array, 
            subtracted_data=subtracted_array,
            target_bin=target_bin, 
            start_frame=start_frame, 
            num_frames=num_frames,
            save_dir=save_directory
        )
        
    print("\n--- ALL TASKS COMPLETED SUCCESSFULLY ---")