
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
import math
import mmap
import DataManagment as dm
from DataManagment import DataPointCollection, DataView
from scipy import optimize as spo
from scipy import integrate as spi
from scipy import signal as sps
from scipy import interpolate as spint
import torch
from scipy import ndimage as spd
from functools import partial
import concurrent.futures
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import itertools

CUDA = False
if torch.cuda.is_available():
    try:
        import nifty_ls
        CUDA = True
    except:
        CUDA = False
from astropy import timeseries as ats

#per ns
kinetics = {
    'laser' : 0.01,
    'radiative' : 1.0/13,
    'isc_0' : 0.001,
    'isc_1' : 0.083,
    's0' : 0.033,
    's1' : 0.01
    }

mapping = {
        'gs.t0': 0,
        'gs.tp': 1,
        'gs.td': 1,
        'es.t0': 2,
        'es.tp': 3,
        'es.td': 3,
        'ms.i': 4
    }

def plotTimeEvolution(group_id : int, data : DataPointCollection, states : list):
    indicies = data._index_map[data.unique_groups[group_id]]
    time = data.data['Time(ns)'][indicies]
    strain_mod = data.data['gs.strain_modulated.ex'][indicies][0]
    strain = data.data['gs.strain.ex'][indicies][0]
    #strain_string = rf"${strain} \pm {strain_mod}$"
    
    figure,ax = plt.subplots()

    for state in states:
        signal = data.data[state][indicies]
        plt.plot(time,signal,label=state)
    plt.xlabel('Time (ns)')
    plt.ylabel('State population')
    #plt.title(rf"Time evoluation of the ground state with a broadband strain modulation of ${strain_string}$")
    plt.legend()
    plt.draw()

def rate_kinetics(k,t_max=5000, start = [1,0,0,0,0]):
    def deriv(y,t):
        g0,g1,es0,es1,m = y
        dg0 = -k['laser']*g0 + k['radiative']*es0 + k['s0']*m
        dg1 = -k['laser']*g1 + k['radiative']*es1 + k['s1']*m
        des0 = k['laser']*g0 - (k['radiative'] + k['isc_0'])*es0
        des1 = k['laser']*g1 - (k['radiative'] + k['isc_1'])*es1
        dm = k['isc_0']*es0 + k['isc_1']*es1 -(k['s0'] + k['s1'])*m
        return [dg0,dg1,des0,des1,dm]
    t = np.linspace(0,t_max,t_max*10)
    sol = spi.odeint(deriv,start,t)
    return t,sol

def get_val_index(dict, val):
    idx = 0
    for k,v in dict.items():
        if v == val:
            break
        idx += 1
    return idx
def get_key_index(dict,key):
    idx = 0
    for k,v in dict.items():
        if k == key:
            break
        idx += 1
    return idx


def identify_elec_state(states : list, elec_states = ['t0', 'tp', 'td', 'i'], energy_states = ['gs', 'es', 'ms'], block = True):
    elec_state = []
    for s in states:
        if s == 'gs.t0' or s == 'es.t0':
            elec_state.append(".")
            continue
        el_state = ""
        es_state = ""
        for s2 in elec_states:
            if s2 in s:
                el_state = s2
                break
        for s2 in energy_states:
            if s2 in s:
                es_state = s2
                break
        new_state = es_state + "." + el_state
        if new_state not in elec_state:
            elec_state.append(es_state + "." + el_state)
        elif block == False:
            elec_state.append(es_state + "." + el_state)
    return elec_state

def plot_kinetics(states = [], sol = None, t = None):
    if sol == None:
        t,sol = rate_kinetics(kinetics)
    states = identify_elec_state(states)
    #print(states)
    

    fig,ax = plt.subplots()
    for i, state_name in enumerate(states):
        if state_name in mapping:
            idx = mapping[state_name]
            plt.plot(t, sol[:, idx], label=f"Kinetics: {states[i]}", linestyle='--')
        else:
            print(f"Warning: State {state_name} not found in kinetics mapping.")

    plt.xlabel('Time (ns)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Kinetic Model Comparison')
    plt.show()


def fft(data : DataPointCollection, state, group_idx):
    indicies = data._index_map[data.unique_groups[group_idx]]
    t = data.data["Time(ns)"][indicies]
    y = data.data[state][indicies]
    f,p = lombscargle2(t,y)
    fig,ax = plt.subplots()

    ax.plot(f,p)
    plt.show()





def plot_standard_deviation(data: DataPointCollection, state, group_idx, window = 100):
    indicies = data._index_map[data.unique_groups[group_idx]]
    t = data.data["Time(ns)"][indicies]
    y = data.data[state][indicies]
    num_steps = len(t)
    dts = np.concatenate([[0],np.diff(t)])
    
    ewma = dm.ExponentiallyWeightedMovingAverage(window)
    vec_update = np.frompyfunc(ewma.update, 2,4)
    s,m,ss,ms = vec_update(y,dts)
    std_devs = s.astype(float)
    std_devs_sig = ss.astype(float)
    means = m.astype(float)
    means_sig = ms.astype(float)
    #for i in range(1,num_steps):
    #    dt = t[i]-t[i-1]
    #    val = y[i]
    #    sigma,mean,sigma_sig, mean_sig = ewma.update(val,dt)
    #    std_devs.append(sigma)
    #    means.append(mean)
    #    means_sig.append(mean_sig)
    #    std_devs_sig.append(sigma_sig)

    
    fig,ax = plt.subplots()
    t_plot = t[0:num_steps]
    ax.plot(t_plot,std_devs, label = rf"$\sigma_{{{state}}}$")
    ax.plot(t_plot,means, label = rf"$\langle{{{state}}}\rangle$")
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('State population')
    ax.legend()
    #ax.vlines([window,2*window,3*window,4*window,5*window],[0,0,0,0,0],[1,1,1,1,1])
    ax.axvspan(0,5*window, color = (117/255,124/255,136/255,0.5))
    plt.draw()
    plt.savefig("mean")

    fig2,ax2 = plt.subplots()
    t_plot = t[0:num_steps]
    ax2.plot(t_plot,std_devs_sig, label = rf"$\sigma_{{\sigma_{{{state}}}}}$")
    #ax2.plot(t_plot,means_sig)
    #ax.vlines([window,2*window,3*window,4*window,5*window],[0,0,0,0,0],[1,1,1,1,1])
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('State population')
    ax2.legend()
    ax2.axvspan(0,100*window, color = (117/255,124/255,136/255,0.5))
    plt.show(block=False)
    plt.savefig("sig")
    return t_plot, means, std_devs, std_devs_sig

def Plot_T_on_TimeEvo(t_data, y_data, T_Calc_func : function):
    T, offset, SNR, freqs= T_Calc_func()
    t_dat = np.array(t_data)
    y = np.array(y_data)
    #T = 11548#popt[1]
    C = 0.33#np.mean(y[-int(len(y)*0.1):])
    A = (y[0]- C)
    #std = cov[1][1]
    fit_func = A * np.exp(-(t_dat-t_dat[0]) / T) + C
    fig, ax = plt.subplots()
    ax.plot(t_dat,y)
    ax.plot(t_dat,fit_func,'r--', label =rf"${A:.2f} \exp \left( \frac{{-t}}{{{T:.2f}}} \right) + {C:.2f}$")
    ax.legend()
    plt.draw()
    plt.show()
    plt.savefig("fig1")
    return T,A,C, SNR, freqs

nnls_solver = LinearRegression(positive=True, fit_intercept=False, copy_X=False)
def solve_single_alpha(alpha, M, y_shifted, n_modes):
    # Construct the Tikhonov-augmented matrices
    M_reg = np.vstack([M, alpha * np.eye(n_modes)])
    y_reg = np.concatenate([y_shifted, np.zeros(n_modes)])
    
    # Solve NNLS
    print(f"{alpha} : Starting")
    #amps, _ = nnls(M_reg, y_reg)

    nnls_solver.fit(M_reg, y_reg)
    amps = nnls_solver.coef_
    
    # Calculate Norms
    res_norm = np.linalg.norm(np.dot(M, amps) - y_shifted)
    sol_norm = np.linalg.norm(amps)
    print(f"{alpha} : Done")
    
    return res_norm, sol_norm, amps

def find_Lcurve_minima(alpha_values, residual_norms, solution_norms):
    x = np.log10(residual_norms)
    y = np.log10(solution_norms)

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(1.5)
    
    optimal_idx = np.argmax(np.abs(curvature))
    return alpha_values[optimal_idx], optimal_idx

def Calculate_T_relaxation(t_data, y_data, start_point):
    t_data = np.asarray(t_data)
    y_data = np.asarray(y_data)
    cuttoff = start_point
    mask = t_data > cuttoff
    #mask_2 = t_data > t_data[-1] - 1000
    t_data = t_data[mask]
    y_data_2 = y_data[mask]
    y_max = np.max(y_data_2)
    y_max_index = np.argmax(y_data_2)
    t_index = t_data[y_max_index]
    mask = t_data > t_index
    y_data_3 = y_data_2[mask]
    t_data_2 = t_data[mask]
    y_data_shifted = y_data_3 - 1/3
    #t_data = t_data[mask]
    #tail = y_data[mask_2]

    modes = 50

    offset = t_data_2[0]
    t_shifted = t_data_2 - offset
    t_shifted = t_shifted / 1e3
    tau_grid = np.linspace(-3,5.5,modes)#np.linespace(-4,1,modes)

    residual_norms = []
    solution_norms = []
    all_amplitudes = []
    alphas = []

    M = np.exp(-t_shifted[:, np.newaxis] / np.power(10, tau_grid))

    alpha = 0.01
    step_factor_bounds = [1.1,5]
    max_alphas = 100

    print("Starting adaptive gradient-controlled L-curve trace")
    while len(alphas) < max_alphas:
        res_norm, sol_norm, amps = solve_single_alpha(alpha, M, y_data_shifted, modes)
        residual_norms.append(res_norm)
        solution_norms.append(sol_norm)
        all_amplitudes.append(amps)
        alphas.append(alpha)

        if len(alphas) >= 2:
            delta_res = (residual_norms[-1] - residual_norms[-2]) / residual_norms[-2]
            total_growth = (residual_norms[-1] - residual_norms[0]) / residual_norms[0]
            delta_res = abs(delta_res)
            total_growth = abs(total_growth)
            if total_growth > 0.005:
                print(f"Early stopping triggered at \u03b1 = {alpha:.4f}! Curve cleared the elbow.")
                break
            sensitivity = 50.0 
            adaptive_multiplier = step_factor_bounds[1] / (1.0 + sensitivity * delta_res)
            step_factor = max(step_factor_bounds[0], min(step_factor_bounds[1], adaptive_multiplier))
        else:
            step_factor = step_factor_bounds[0]
        alpha *= step_factor

    print("Calculation finished successfully!")
    fig,ax = plt.subplots(figsize=(7,6))
    ax.plot(residual_norms, solution_norms, '-o', markersize=4, color='blue', label='L-curve')
    ax.set_yscale('log')

    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    plt.xticks(rotation=25)

    min_spatial_distance = 0.12 
    last_x, last_y = float('-inf'), float('-inf')
    x = residual_norms 
    y_log = np.log10(solution_norms)
    for i in range(len(alphas)):
        distance = np.sqrt((x[i] - last_x)**2 + (y_log[i] - last_y)**2)
        if distance >= min_spatial_distance:
            plt.annotate(
                f"α={alphas[i]:.4f}", 
                (residual_norms[i], solution_norms[i]),
                textcoords="offset points", 
                xytext=(10, 5), # Constant small offset from the dot
                ha='left', 
                fontsize=9
            )
            last_x, last_y = x[i], y_log[i]
        
    plt.xlabel('Residual Norm ||M*A - y||₂ (Fit Error)')
    plt.ylabel('Solution Norm ||A||₂ (Spectrum Structure)')
    plt.title('L-Curve for Finding Optimal Regularization')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.axvspan(min(residual_norms), residual_norms[-2], color='green', alpha=0.1, label='Elbow Region')
    plt.legend()
    plt.show()
    plt.savefig('alpha')

    fig,ax = plt.subplots(figsize=(7,6))
    ax.plot(alphas, residual_norms,'-o',markersize=4,color='red')
    plt.ylabel('Residual Norm ||M*A - y||₂ (Fit Error)')
    plt.xlabel(f"α")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig('alpha_residual')

    opt_idx = len(alphas)-2
    alpha_opt = alphas[opt_idx]
    best_amplitudes = all_amplitudes[opt_idx]

    mode_weights = best_amplitudes

    actual_taus = np.power(10, tau_grid) * 1e3

    total = np.sum(mode_weights)
    factor = 1 / total
    normalized_weights = mode_weights * factor
    threshold = 0.05
    mask = normalized_weights > threshold
    weights = normalized_weights[mask]
    if(np.sum(weights) > 0):
        avg_tau = np.sum(weights * actual_taus[mask]) / np.sum(weights)
    else:
        avg_tau = 0.0
    

    #mask = mode_weights > 0.005
    #if np.sum(mode_weights[mask]) > 0:
    #    avg_tau = np.sum(mode_weights[mask] * actual_taus[mask]) / np.sum(mode_weights[mask])
    #else:
    #    avg_tau = 0  
    print(f"Ensemble Average Decay Time (Tau): {avg_tau:.4f}")

    
    t_shifted = t_data - t_data[0]
    exponentials_fit = np.exp(-t_shifted[:, np.newaxis] / actual_taus)
    y_fit_line = 1/3 + (exponentials_fit @ mode_weights)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(t_data, y_data_2, color='black', alpha=0.5, label='')
    plt.plot(t_shifted, y_fit_line, color='crimson', lw=2.5, label='50-Mode Log-Lifetime Reconstruction')
    plt.axvspan(0, cuttoff, color='gray', alpha=0.15, label='Ignored Transient Window')
    plt.xlabel('Time (ns)', fontsize=11)
    plt.ylabel('Mean', fontsize=11)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("relaxation.png")
    plt.close()

    plot_residuals(y_data_2,y_fit_line,t_data)
    return avg_tau, offset, GetSNR(y_data_2,y_fit_line), classify_bath(y_data_2,y_fit_line,t_data)

def residuals(raw,fit):
    residual = raw - fit
    points = len(raw)
    noise = np.sum(residual**2) / points
    return noise

def plot_residuals(raw,fit,t_data):
    residuals = abs(raw-fit)
    fig,ax = plt.subplots()
    ax.plot(t_data,residuals,color='crimson',lw=2.5)
    fig.savefig('residuals')
    plt.show()

def classify_bath(raw, fit,t_data):
    total_time = t_data[-1] - t_data[0]
    points = len(t_data)
    mean_dt = total_time / points
    fs_uniform = 1 / mean_dt
    t_uniform = np.linspace(t_data[0], t_data[-1], points)

    cum_max = np.maximum.accumulate(t_data)
    strict_mask = np.zeros(len(t_data), dtype=bool)
    strict_mask[0] = True
    strict_mask[1:] = t_data[1:] > cum_max[:-1]

    raw_res = raw-fit

    t = t_data[strict_mask]
    res_clean = raw_res[strict_mask]

    interp_func = spint.interp1d(t, res_clean, kind='cubic')
    res = interp_func(t_uniform)
    freq,psd = sps.welch(res,fs=fs_uniform,nperseg=1024)

    fig,ax = plt.subplots()
    ax.plot(freq,psd)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("freq")
    plt.show()
    plt.close()
    return [freq,psd]
    
    #peaks,properties = signal.find_peaks(psd,prominence=np.mean(psd)*0.1)
    #if len(peaks) == 0:
    #    return "Overdamped / Decoupled (Pure Noise Floor)"
    #
    #largest_peak_idx = peaks[np.argmax(psd[peaks])]
    #peak_power = psd[largest_peak_idx]
    #median_noise = np.median(psd)
    #
    #pnr = peak_power / median_noise
    #prominence_threshold = 5.0
#
    #if pnr > prominence_threshold:
    #    return f"Underdamped (Coherent Ringing at {freq[largest_peak_idx]:.4f} Hz)"
    #else:
    #    return "Overdamped / Thermalized (No Dominant Coherent Mode)"



def signal(raw):
    points = len(raw)
    power = np.sum(raw**2) / points
    return power

def GetSNR(raw,fit):
    raw_dyn = raw - np.min(raw)
    noise = residuals(raw_dyn,fit)
    snr = (signal(raw_dyn))/noise
    snr_db = 10 * np.log10(snr)
    return snr_db

def get_time_index(t_data,time):
    idx = np.searchsorted(t_data, time)
    if idx < len(t_data):
        return idx
    else:
        return len(t_data)

def generate_strain_frequency_map(strain_list, freq_list, pow_list):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.25,
        "grid.linestyle": "--"
    })

    powers = []
    for p in pow_list:
        #powers.append(np.log10(p))
        powers.append(p)

    frequencies = freq_list[0]
    frequencies_safe = np.where(frequencies == 0, 1e-5, frequencies)
    log_strain = np.log10(strain_list)
    log_freq = np.log10(frequencies_safe)
    PSD_grid = np.column_stack(powers)
    PSD_dB = 10 * np.log10(np.clip(PSD_grid, 1e-15, None))
    interp_func = spint.RegularGridInterpolator((log_freq, log_strain), PSD_dB, method='cubic', bounds_error=False, fill_value=None)
    
    dense_log_strain = np.linspace(log_strain.min(), log_strain.max(), 500)
    dense_log_freq = np.linspace(log_freq.min(), log_freq.max(), 500)
    Dense_Log_Strain_Mesh, Dense_Log_Freq_Mesh = np.meshgrid(dense_log_strain, dense_log_freq)

    dense_points = np.vstack([Dense_Log_Freq_Mesh.ravel(), Dense_Log_Strain_Mesh.ravel()]).T
    PSD_dB_dense = interp_func(dense_points).reshape(Dense_Log_Strain_Mesh.shape)   

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    mesh = ax.pcolormesh(strain_list, frequencies_safe, PSD_dB, 
                     shading='gouraud', 
                     cmap='magma',
                     vmin=-115, vmax=-20)


    valid_data = PSD_dB_dense[PSD_dB_dense > -1000].flatten()
    min_val = np.min(valid_data)
    q25 = np.percentile(valid_data, 25)
    p_low = 1.5 if (q25 - min_val) > 20 else 5.0
    p_mid1 = 25.0
    p_mid2 = 55.0
    q75 = np.percentile(valid_data, 75)
    max_val = np.max(valid_data)
    p_high = 92.0 if (max_val - q75) < 15 else 85.0
    dynamic_percentiles = [p_low, 15.0, 35.0, 55.0, 72.0, p_high]
    contour_levels = np.percentile(valid_data, dynamic_percentiles)
    print(f"Automatically generated contour levels for this file: {np.round(contour_levels, 1)} dB")
    #contour_levels = np.arange(PSD_dB.min(), PSD_dB.max(), 15)
    contours = ax.contour(10**Dense_Log_Strain_Mesh, 10**Dense_Log_Freq_Mesh, PSD_dB_dense, 
                      levels=contour_levels, 
                      colors="#000000ff",    
                      linewidths=0.7,      
                      alpha=0.75)

    ax.set_xscale('log')  
    ax.set_yscale('log')  
    
    ax.set_xlim(np.min(strain_list), np.max(strain_list))
    ax.set_ylim(5e-4, 0.5)

    #ax.set_xlabel('Applied Strain ($\epsilon$)', fontweight='bold', labelpad=8)
    #ax.set_ylabel('Frequency Domain ($f$)', fontweight='bold', labelpad=8)

    ax.grid(True, which="both", color="white", alpha=0.15)

    plt.xlabel('Applied Strain ($\epsilon$)', fontweight='bold')
    plt.ylabel('Frequency ($f$, GHz)', fontweight='bold')
    plt.title('Decibel ($10 \log_{10}$ dB) Power Spectrogram', fontweight='bold')
    cbar1 = plt.colorbar(mesh)
    cbar1.set_label('Relative Power Intensity (dB)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('spectrogram_decibels.png', dpi=300)
    plt.clf()

    #-------------------------------------------------------------#

    PSD_norm = np.zeros_like(PSD_grid)

    for i in range(PSD_grid.shape[1]):
        col_min = PSD_grid[:, i].min()
        col_max = PSD_grid[:, i].max()
        if col_max > col_min:
            PSD_norm[:, i] = (PSD_grid[:, i] - col_min) / (col_max - col_min)

    Y_freq_safe = np.where(frequencies == 0, 1e-5, frequencies)
    
    pcm2 = plt.pcolormesh(strain_list, Y_freq_safe, PSD_norm, 
                        shading='auto', 
                        cmap='magma', 
                        vmin=0.0, vmax=1.0) # Scale is now locked strictly between 0 and 1

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Applied Strain ($\epsilon$)', fontweight='bold')
    plt.ylabel('Frequency ($f$, GHz)', fontweight='bold')
    plt.title('Column-Normalized Structural Phase Map', fontweight='bold')

    cbar2 = plt.colorbar(pcm2)
    cbar2.set_label('Normalized Relative Intensity (0.0 to 1.0)', fontweight='bold')

    plt.tight_layout()
    plt.savefig('spectrogram_normalized_phase.png', dpi=300)
    plt.clf()

    #cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    #cbar.set_label('Spectral Power Intensity (Residual Coherence)', 
                   #fontweight='bold', labelpad=10)
    
    # 7. Add Regime Labels directly onto the map for presentation clarity
    #ax.text(1.5e-5, 1e-1, 'I. Frozen\nRegime', color='white', alpha=0.7,
    #        fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.5'))
    
    #ax.text(4e-4, 1.5e-2, 'II. Underdamped\nCoherent Ringing\n(Modal Splintering)', color='black',
    #        fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), ha='center')
    
    #ax.text(6e-3, 1e-1, 'III. Overdamped\nDissipation', color='white', alpha=0.7,
    #        fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.5'))

    #plt.tight_layout()
    #plt.savefig('strain_frequency_power_spectrogram-2.png', dpi=300)
    #plt.show()

def process_single_file(i, dat, window):
    try:
        # 1. Compute standard deviation window metrics
        t, m, sd, sdsd = plot_standard_deviation(dat, 'gs.t0', i, window)
        
        # 2. Get data components for this specific unique group
        indices = dat._index_map[dat.unique_groups[i]]
        time = dat.data['Time(ns)'][indices]
        signal = dat.data['gs.t0'][indices]
        strain = dat.data['gs.strain.ex'][indices][0]
        
        # 3. Fit relaxation and compute spectrum
        bound_T = partial(Calculate_T_relaxation, t, m, 10*window)
        T, _, C, sn, f = Plot_T_on_TimeEvo(time, signal, bound_T)
        
        return {
            'strain': strain,
            'T': T,
            'C': C,
            'snr': sn,
            'freq': f[0],
            'fpow': f[1]
        }
    except Exception as e:
        print(f"Error processing index {i}: {e}")
        return None


def main(load : bool = False):

    #base_path = "../GitHub/MolSpin/NV_Centre_N14_results/"
    #base_file_name = "NV_centre_N14_GS-T"
    #base_file = base_path + base_file_name
    #dm.BatchProcess(base_file_name, 0,2, base_path=base_path)
   # return

    #file_name = "NV_centre_N14_GS-T1-3"
    #file_name = "NV_centre_N14_GS-T1-4"
    #file_name = "NV_centre_N14_GS-T1-5"
    #file_name = "NV_centre_N14_GS-T1-29"
    #file_name = "NV_centre_N14_GS-T1-22"
    file_name = "NV_centre_N14_GS-T-"
    #file_dict = "../GitHub/MolSpin/NV_Centre_N14_results/"
    #file_dict_ssh = "Documents/GitHub/MolSpin/NV_Centre_N14_results/"
    #extension = "-4"
    extension = "8"
    filenames = [file_name + rf"{i}" + ".npz" for i in range(0,10)]
    
    npzfile = ""
    #file = file_dict + file_name# + extension
    file = file_name + extension
    if(not load):
        npzfile = dm.ProcessFile(file,file_name+extension)
        #npzfile = dm.ProcessFileSSH(file_name, file_name,file_dict_ssh,"scandium.qbl.uni-oldenburg.de","juft2450")
    else:
        npzfile = file + ".npz"
    #dat = DataPointCollection(npzfile)
    #dat = DataPointCollection(file_name + extension + ".npz")
    dat = DataPointCollection(filenames)

    states = [['gs.t0_u', 'gs.t0_z', 'gs.t0_d', 'gs.tp_u', 'gs.tp_z', 'gs.tp_d', 'gs.td_u', 'gs.td_z', 'gs.td_d'],
              ['es.t0_u', 'es.t0_z', 'es.t0_d', 'es.tp_u', 'es.tp_z', 'es.tp_d', 'es.td_u', 'es.td_z', 'es.td_d'],
              ['ms.i'] ]
    #plotTimeEvolution(0,dat,['gs.t0'])
    #plotTimeEvolution(50,dat,states[0])
    #plotTimeEvolution(92,dat,states[0])
    #plt.show(block = False)

    #plotT1(dat,['gs.t0_u'])
    #plt.show(block = False)
    #plotT2(dat,['gs.t0_u'])#, 'gs.t0_z', 'gs.t0_d'])
    #plt.show()
    #P,I = find_T1(0,dat,'gs.t0')
    #print(P[1])
    #plotT1onTimeEvoCurve(0,dat,'gs.t0')
    #fft(dat,'gs.t0', 0)
    #plot_peak_decay(dat,'gs.t0', 0)
    #plot_standard_deviation(dat[0:int(len(dat)/4)],'gs.t0', 0,100)
    window = 100
    warm_up = 5*window
    regime = ["slow","slow", "slow", "slow", "slow", "normal", "fast", "fast", "fast", "fast"]
    T_times = []
    x = []
    C_val = []
    snr = []
    freq = []
    fpow = []

    indices_to_process = range(6, len(filenames))
    worker_fn = partial(process_single_file, dat=dat, window=window)
    for i in indices_to_process:
        result = worker_fn(i)
        if result is not None:
            # Append the calculated values back to your master arrays
            x.append(result['strain'])
            T_times.append(result['T'])
            C_val.append(result['C'])
            snr.append(result['snr'])
            freq.append(result['freq'])
            fpow.append(result['fpow'])


    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #    worker_fn = partial(process_single_file, dat=dat, window=window)
    #    results = executor.map(worker_fn, indices_to_process)
#
    #    for result in results:
    #        if result is not None:
    #            # Append the calculated values back to your master arrays
    #            x.append(result['strain'])
    #            T_times.append(result['T'])
    #            C_val.append(result['C'])
    #            snr.append(result['snr'])
    #            freq.append(result['freq'])
    #            fpow.append(result['fpow'])
#
    #print(f"Successfully processed {len(x)} files in parallel!")

    #for i in range(5,len(filenames)):
    #    t, m, sd, sdsd = plot_standard_deviation(dat,'gs.t0', i,window)
    #    #time_idx = get_time_index(t,warm_up)
#
    #    bound_T = partial(Calculate_T_relaxation,t,m,1000)
    #    indicies = dat._index_map[dat.unique_groups[i]]
    #    time = dat.data['Time(ns)'][indicies]
    #    signal = dat.data['gs.t0'][indicies]
    #    strain = dat.data['gs.strain.ex'][indicies][0]
    #    T, _,C,sn,f = Plot_T_on_TimeEvo(time,signal, bound_T)
    #    T_times.append(T)
    #    C_val.append(C)
    #    x.append(strain)
    #    snr.append(sn)
    #    freq.append(f[0])
    #    fpow.append(f[1])
    
    fig,ax = plt.subplots()
    #ax.errorbar(x,T_times,yerr=std, fmt='o', capsize=5, capthick=1, color='blue',ecolor='red')
    ax.scatter(x,T_times,color='blue')
    ax.set_yscale('log')
    ax.set_xscale('log')
    strain = x
    log_y = np.log(T_times)
    log_x = np.log(x)
    slope,intercept = np.polyfit(log_x,log_y,1)
    a = np.exp(intercept)
    b = slope
    x = np.array(x)
    func = lambda x, a, b : a * np.exp(b*x)
    x_fit = np.geomspace(min(x),max(x), 1000)
    fit = a * (x_fit**b) 
    ax.plot(x_fit,fit,linestyle='--',color = 'black', linewidth=2)
    plt.show(block=True)
    plt.savefig("T-relaxation.png")

    fig,ax = plt.subplots()
    ax.plot(strain,snr, '-o',color='red')
    ax.set_xscale('log')
    plt.show(block=False)
    plt.savefig("SNR-ratio")

    generate_strain_frequency_map(strain,freq,fpow)




    

    #peak_


    t,sol = rate_kinetics(kinetics)

    #plot_kinetics(states[0] + states[1] + states[2])


if __name__ == "__main__":
    #main()
    main(True)