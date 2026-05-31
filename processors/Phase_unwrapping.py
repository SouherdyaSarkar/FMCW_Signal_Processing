import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from configuration import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from openradar.mmwave.dataloader import DCA1000
from openradar.mmwave import dsp
from openradar.mmwave.dsp.utils import Window
import time
from datetime import datetime
import sys

DATA_PATH="data/adc_data_2026-05-26_15-17-06_phone_normal.npy"
numFrames = 200
numADCSamples = 256
numTxAntennas = 3
numRxAntennas = 4
numLoopsPerFrame = 182
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

stop_flag = False
start_flag = True  
Tp = 14e-6
Tc = 72e-6

def init_dca():
    dca = DCA1000()
    return dca 

def collect_data(dca, num_frames=1):
    adc_data = dca.read(num_frames=int(num_frames))
    return adc_data

def stop_plot(event):
    global stop_flag, start_flag
    stop_flag = True
    start_flag = False

def start_plot(event):
    global stop_flag, start_flag
    stop_flag = False
    start_flag = True

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

def iterative_range_bins_detection(rangeResult):
    rangeResult = np.transpose(np.stack([rangeResult[0::3], rangeResult[1::3], rangeResult[2::3]], axis=1),axes=(1,2,0,3))
    range_result_absnormal_split=[]
    for i in range(numTxAntennas):
        for j in range(numRxAntennas):
            r_r=np.abs(rangeResult[i][j])
            #first 10 range bins i.e 40 cm make it zero
            r_r[:,0:10]=0
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            r_r_normalise = (r_r - min_val) / (max_val - min_val) * (1000 - 0) + 0
            range_result_absnormal_split.append(r_r_normalise)
    
    range_abs_combined_nparray=np.zeros((numLoopsPerFrame,numADCSamples))
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray+=ele
    range_abs_combined_nparray/=(numTxAntennas*numRxAntennas)
    
    range_abs_combined_nparray_collapsed=np.sum(range_abs_combined_nparray,axis=0)/numLoopsPerFrame
    peaks_min_intensity_threshold = np.argsort(range_abs_combined_nparray_collapsed)[::-1][:5]
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
    

if __name__ == "__main__":
    plt.ion()
    fig = None
    i = 0 
    prev_range_bins = None
    overlapped_range_bins = []

    loaded_adc_data = np.load(DATA_PATH)
    while i < 100 and i < len(loaded_adc_data):
        current_frame = loaded_adc_data[i:i+1]
        adc_data = np.apply_along_axis(DCA1000.organize, 1, current_frame, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)
        radar_cube = dsp.range_processing(adc_data[0], window_type_1d=Window.BLACKMAN)
        
        # Calculate Range FFT
        rangefft_out = np.abs(radar_cube).sum(axis=(0,1))
        # max_range_index, range_bins, rangeResult = iterative_range_bins_detection(radar_cube)
        # target_bin = range_bins[0] if len(range_bins) > 0 else 0
        # unwrapped_phase = get_phase_antennawise(rangeResult[0][0], target_bin)

        if fig is None:
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(1, 1, 1) 

            ax_stop = fig.add_axes([0.75, 0.92, 0.1, 0.05])
            ax_start = fig.add_axes([0.87, 0.92, 0.1, 0.05])
            btn_stop = Button(ax_stop, 'Stop')
            btn_start = Button(ax_start, 'Start')
            btn_stop.on_clicked(stop_plot)
            btn_start.on_clicked(start_plot)

        ax1.cla()

        # --- Range FFT Plotting ---
        ax1.plot(rangefft_out, color='blue')
        ax1.set_title(f"1D Range Profile (FFT) | Frame: {i+1}")
        ax1.set_xlabel("Range Bins")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        plt.pause(0.1)

        i += 1  

    plt.ioff()
    plt.show()
