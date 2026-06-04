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
import sys
import math

DATA_PATH="data/Mobile_data/adc_data_2026-06-03_16-05-58_pho_vib.npy"
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

    # Get indices of detected peaks
    full_mask = (det_doppler_mask & det_range_mask)
    det_peaks_indices = np.argwhere(full_mask == True)

    # peakVals and SNR calculation
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

    # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
    detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

    # --- Peak Grouping
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
            
            r_r[:, :min_bin] = 0 # Zero out everything before min_bin
            if max_bin is not None:
                r_r[:, max_bin:] = 0 # Zero out everything after max_bin
            # ----------------------------------------------
                
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            
            # Prevent division by zero if the array is completely empty
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

def solve_equation(phase_cur_frame):
    phase_diff=[]
    for soham in range (1,len(phase_cur_frame)):
        phase_diff.append(phase_cur_frame[soham]-phase_cur_frame[soham-1])
    L=100
    r0=20
    roots_of_frame=[]
    for i,val in enumerate(phase_diff):
        c=(phase_diff[i]*0.001/3.14)/(3*(Tp+Tc))
        t=3*(i+1)*(Tp+Tc)
        c1=t*t
        c2=-2*L*t
        c3=L*L-c*c*t*t
        c4=2*L*c*c*t
        c5=-r0*r0*c*c
        coefficients=[c1, c2, c3, c4, c5]
        root=min(np.abs(np.roots(coefficients)))
        roots_of_frame.append(root)
    median_root=np.median(roots_of_frame)
    final_roots=[]
    for root in roots_of_frame:
        if root >0.9*median_root and root<1.1*median_root:
            final_roots.append(root)
    return np.mean(final_roots)

def get_velocity_antennawise(range_FFT_,peak):
    phase_per_antenna=[]
    vel_peak=[]
    for k in range(0,numLoopsPerFrame):
        r = range_FFT_[k][peak].real
        i = range_FFT_[k][peak].imag
        phase=get_phase(r,i)
        phase_per_antenna.append(phase)
    phase_cur_frame=phase_unwrapping(len(phase_per_antenna),phase_per_antenna)
    cur_vel=solve_equation(phase_cur_frame)
    return cur_vel

def get_phase_antennawise(range_FFT_,peak):
    phase_per_antenna=[]
    vel_peak=[]
    for k in range(0,numLoopsPerFrame):
        r = range_FFT_[k][peak].real
        i = range_FFT_[k][peak].imag
        phase=get_phase(r,i)
        phase_per_antenna.append(phase)
    phase_cur_frame=phase_unwrapping(len(phase_per_antenna),phase_per_antenna)
    return phase_cur_frame

def get_velocity(rangeResult,range_peaks):
    vel_array_frame=[]
    for peak in range_peaks:
        vel_arr_all_ant=[]
        for i in range(0,numTxAntennas):
            for j in range(0,numRxAntennas):
                cur_velocity=get_velocity_antennawise(rangeResult[i][j],peak)
                vel_arr_all_ant.append(cur_velocity)
        vel_array_frame.append(vel_arr_all_ant)
    return vel_array_frame

def dopplerFFT(rangeResult):  #
    windowedBins2D = rangeResult * np.reshape(np.hamming(numLoopsPerFrame), (1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult

def speed_estimation_fn(range_bins, rangeResult):
    vel_array_frame = np.array(get_velocity(rangeResult,range_bins)).flatten()
    return vel_array_frame
    
def get_averaged_phase(rangeResult, target_bin):
    all_phases = []
    for tx in range(numTxAntennas):
        for rx in range(numRxAntennas):
            unwrapped_phase = get_phase_antennawise(rangeResult[tx][rx], target_bin)
            all_phases.append(unwrapped_phase)
    all_phases = np.array(all_phases)
    avg_phase = np.mean(all_phases, axis=0)
    
    return avg_phase

def get_and_plot_phase_differences(rangeResult, target_bin, frame_no, save_dir="Simulations/Mobile_Vibration_Phase"):
    avg_phase = get_averaged_phase(rangeResult, target_bin)
    
    phase_diff = np.insert(np.diff(avg_phase), 0, 0)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(phase_diff, color='red', marker='.')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)

    max_val = np.max(np.abs(phase_diff))
    ax.set_ylim(-0.5,0.5)
    
    ax.set_title(f"Phase Difference - Range Bin: {target_bin}")
    ax.set_xlabel("Chirp Index")
    ax.set_ylabel("Phase Diff (Rad)")
    ax.grid(True)
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"Simulations/Phone_data_day02/Phone_vibration_phases/frame{frame_no}_{target_bin}_phase_diff.png")
    plt.close(fig)
    
    return phase_diff

def save_averaged_phase_plots(rangeResult, selected_bins, frame_no):
    save_dir = "Simulations/Mobile_Vibration_Phase"
    os.makedirs(save_dir, exist_ok=True)
    max_per_plot = 10
    chunks = [selected_bins[i:i + max_per_plot] for i in range(0, len(selected_bins), max_per_plot)]

    for chunk_idx, chunk in enumerate(chunks):
        num_bins = len(chunk)
        cols = 2 if num_bins > 5 else 1
        rows = 5
        fig_width = 15 if cols == 2 else 10
        fig_stacked, axes = plt.subplots(rows, cols, figsize=(fig_width, 15), sharex=True)
        
        title_suffix = f" (Part {chunk_idx + 1})" if len(chunks) > 1 else ""
        fig_stacked.suptitle(f"Averaged Unwrapped Phase{title_suffix}", fontsize=16, fontweight='bold')
        axes_flat = axes.flatten()
        
        for idx, target_bin in enumerate(chunk):
            avg_phase = get_averaged_phase(rangeResult, target_bin)
            
            ax = axes_flat[idx]
            ax.plot(avg_phase, color='purple', marker='.')
            ax.set_title(f"Range Bin: {target_bin}")
            ax.set_ylabel("Phase (Rad)")
            ax.grid(True)

            # Create, Save, and Close Individual Figure
            fig_ind, ax_ind = plt.subplots(figsize=(8, 4))
            ax_ind.plot(avg_phase, color='green', marker='.')
            ax_ind.set_title(f"Averaged Phase - Range Bin: {target_bin}")
            ax_ind.set_xlabel("Chirp Index")
            ax_ind.set_ylabel("Phase (Rad)")
            ax_ind.grid(True)
            
            fig_ind.tight_layout()
            fig_ind.savefig(os.path.join(save_dir, f"frame{frame_no}_{target_bin}_avg_phase.png"))
            plt.close(fig_ind) 

        for idx in range(num_bins, len(axes_flat)):
            fig_stacked.delaxes(axes_flat[idx])

        for i in range(1, cols + 1):
            if num_bins - i >= 0:
                axes_flat[num_bins - i].set_xlabel("Chirp Index")
                axes_flat[num_bins - i].tick_params(labelbottom=True) 
                
        fig_stacked.tight_layout()
        file_suffix = f"_part{chunk_idx + 1}" if len(chunks) > 1 else ""
        fig_stacked.savefig(os.path.join(save_dir, f"frame{frame_no}_stacked_avg_phase{file_suffix}.png"))
        plt.close(fig_stacked)
        
    print(f"Saved {len(selected_bins)} individual plots and {len(chunks)} stacked plot(s) to {os.path.abspath(save_dir)}")

def save_phase_plots(range_result_antenna, selected_bins, frame_no):
    save_dir = "Simulations/Mobile_Phase_simulations"
    os.makedirs(save_dir, exist_ok=True)

    # Setup Stacked Figure (5 rows, 2 columns)
    fig_stacked, axes = plt.subplots(5, 2, figsize=(15, 15), sharex=True)
    fig_stacked.suptitle("Unwrapped Phase for Selected Range Bins", fontsize=16, fontweight='bold')

    for idx, target_bin in enumerate(selected_bins):
        unwrapped_phase = get_phase_antennawise(range_result_antenna, target_bin)

        # Correctly map 1D index to 2D grid (5 rows, 2 cols)
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ax.plot(unwrapped_phase, color='red', marker='.')
        ax.set_title(f"Range Bin: {target_bin}")
        ax.set_ylabel("Phase (Rad)")
        ax.set_ylim(-120,10)
        ax.grid(True)

        # Create, Save, and Close Individual Figure
        fig_ind, ax_ind = plt.subplots(figsize=(8, 4))
        ax_ind.plot(unwrapped_phase, color='blue', marker='.')
        ax_ind.set_title(f"Unwrapped Phase - Range Bin: {target_bin}")
        ax_ind.set_xlabel("Chirp Index")
        ax_ind.set_ylabel("Phase (Rad)")
        ax_ind.grid(True)
        
        fig_ind.tight_layout()
        fig_ind.savefig(os.path.join(save_dir, f"frame{frame_no}_{target_bin}_phase.png"))
        plt.close(fig_ind) 

    # Set x-labels only on the bottom row of the 5x2 grid
    for col in range(2):
        axes[4, col].set_xlabel("Chirp Index")
        
    fig_stacked.tight_layout()
    fig_stacked.savefig(os.path.join(save_dir, f"frame{frame_no}_stacked_phase.png"))
    plt.close(fig_stacked)
    
    print(f"Saved {len(selected_bins)} plots to {os.path.abspath(save_dir)}")

def plot_continuous_phase_diff(all_phase_diffs, target_bin, start_frame, num_frames, save_dir):
    fig, ax = plt.subplots(figsize=(15, 4))
    
    ax.plot(all_phase_diffs, color='red', marker='.', markersize=0.5, linestyle='-', linewidth=0.3, alpha=0.8)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2)
    ax.set_title(f"Continuous Phase Difference - Range Bin: {target_bin}\n(Frames {start_frame} to {start_frame + num_frames})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Cumulative Chirp Index (Time)")
    ax.set_ylabel("Phase Diff (Rad)")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    fig.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"frames{start_frame}_to_{start_frame+num_frames}_bin{target_bin}_continuous.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved continuous plot for bin {target_bin} at {save_path}")

if __name__ == "__main__":
    loaded_adc_data = np.load(DATA_PATH)
    start_frame = 300
    num_frames = 10  
    target_bins = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    accumulated_phase_diffs = {bin_idx: [] for bin_idx in target_bins}
    
    print(f"Processing frames {start_frame} to {start_frame + num_frames}...")
    for f in range(start_frame, start_frame + num_frames + 1):
        current_frame = loaded_adc_data[f-1 : f]
        adc_data = np.apply_along_axis(DCA1000.organize, 1, current_frame, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)
        radar_cube = dsp.range_processing(adc_data[0], window_type_1d=Window.BLACKMAN)
        
        min_b = 0
        max_b = 20
        _, range_bins, rangeResult = iterative_range_bins_detection(radar_cube, min_bin=min_b, max_bin=max_b)
 
        for target_bin in target_bins:
            avg_phase = get_averaged_phase(rangeResult, target_bin)
            phase_diff = np.insert(np.diff(avg_phase), 0, 0)
            accumulated_phase_diffs[target_bin].extend(phase_diff)

    save_directory = "Simulations/Radar/Phone_data_day02/Phone_vibration_phases"
    
    for target_bin in target_bins:
        all_diffs_array = np.array(accumulated_phase_diffs[target_bin])
        plot_continuous_phase_diff(
            all_phase_diffs=all_diffs_array, 
            target_bin=target_bin, 
            start_frame=start_frame, 
            num_frames=num_frames,
            save_dir=save_directory
        )
        
    print("All multi-block plots generated successfully!")
