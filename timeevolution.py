
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
#from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize

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


def lombscargle(t,y):
    y_centre = y - np.mean(y)
    freqs = np.linspace(0.001, 10, 10000)
    pgram = sps.lombscargle(t,y_centre, freqs*2*np.pi)
    return freqs[np.argmax(pgram)]

def lombscargle2(t,y):
    y_centre = y - np.mean(y)
    pgram = []
    freqs = []
    if CUDA:
        freq, power = ats.LombScargle(t,y,nterms=1).autopower(method="fastnifty")
    else:
        freq, power = ats.LombScargle(t,y,nterms=1).autopower()
    #for i in range(5):
    #    freq_i = np.linspace(i, i+1, 1000)
    #    pgram_ = #sps.lombscargle(t,y_centre, freq_i*2*np.pi)
    
    #    freqs.append(freq_i)
    #    pgram.append(pgram_)
    return freq, power



def fft(data : DataPointCollection, state, group_idx):
    indicies = data._index_map[data.unique_groups[group_idx]]
    t = data.data["Time(ns)"][indicies]
    y = data.data[state][indicies]
    f,p = lombscargle2(t,y)
    fig,ax = plt.subplots()

    ax.plot(f,p)
    plt.show()

def peak_decay(data: DataPointCollection, state, group_idx, dist = 10):
    indicies = data._index_map[data.unique_groups[group_idx]]
    t = data.data["Time(ns)"][indicies]
    y = data.data[state][indicies]
    mask = np.diff(t,prepend=t[0]-1) > 1e-15
    t = t[mask]
    y = y[mask]
    peaks, _ = sps.find_peaks(y,distance=dist)
    peak_t = np.concatenate(([t[0]], t[peaks], [t[-1]]))
    peak_y = np.concatenate(([y[0]], y[peaks], [y[-1]]))

    f_env = spint.interp1d(peak_t, peak_y, kind='cubic', fill_value='extrapolate')
    return f_env(t),t,y

def plot_peak_decay(data: DataPointCollection, state, group_idx):
    upper_env,t,y = peak_decay(data,state,group_idx)
    fig,ax = plt.subplots()
    ax.plot(t,y,alpha = 0.4)
    ax.plot(t,upper_env, 'r', linewidth = 2)
    plt.show(block=False)

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
    return t_plot, means, std_devs, std_devs_sig

def Plot_T_on_TimeEvo(t_data, y_data, T_Calc_func : function):
    #popt, line, start, end, cov = T_Calc_func()
    t_dat = np.array(t_data)
    y = np.array(y_data)
    T = 8411.62#popt[1]
    C = np.mean(y[-int(len(y)*0.1):])
    A = y[0]- C
    #std = cov[1][1]
    fit_func = A * np.exp(-(t_dat-t_dat[0]) / T) + C
    fig, ax = plt.subplots()
    ax.plot(t_dat,y)
    ax.plot(t_dat,fit_func,'r--', label =rf"${A:.2f} \exp \left( \frac{{-t}}{{{T:.2f}}} \right) + {C:.2f}$")
    ax.legend()
    plt.draw()
    plt.show()
    return T,A,C#,std

def multi_mode_exp_decay(flat_params,x):
    a = flat_params[:50] / 100
    log_t = flat_params[50:100]
    c = flat_params[-1] / 10
    
    t = np.exp(log_t)
    modes = a[:,np.newaxis] * np.exp(-x/t[:,np.newaxis] )
    sum = np.sum(modes,axis = 0)
    return c + sum

def loss_function(flat_params, t, y, modes = 50):
    y_pred = multi_mode_exp_decay(flat_params,t)
    mean = np.mean((y-y_pred)**2)
    return mean

def jacobian_function(flat_params, x, y):
    # 1. Unpack parameters
    a = flat_params[:50]
    log_t = flat_params[50:100]
    c = flat_params[-1]
    t = np.exp(log_t)
    
    # 2. Get predictions and residuals
    modes = a[:, np.newaxis] * np.exp(-x / t[:, np.newaxis])
    y_pred = c + np.sum(modes, axis=0)
    residual = y_pred - y  # Shape: (N,)
    N = len(x)
    
    # 3. Calculate analytical derivatives using calculus
    # Derivative with respect to amplitudes (a)
    grad_a = (2 / N) * np.dot(np.exp(-x / t[:, np.newaxis]), residual)
    
    # Derivative with respect to log_t
    # d(model)/d(log_t) = a * (x / t) * exp(-x / t)
    inner_t = a[:, np.newaxis] * (x / t[:, np.newaxis]) * np.exp(-x / t[:, np.newaxis])
    grad_log_t = (2 / N) * np.dot(inner_t, residual)
    
    # Derivative with respect to offset (c)
    grad_c = (2 / N) * np.sum(residual)
    
    # Return as a single flat array matching the 101 parameter structure
    return np.concatenate([grad_a, grad_log_t, [grad_c]])

def Calculate_T_relaxation(t_data, y_data, start_point, dist = 5000, mode = False,fast_time = 25000):
    t_data = np.asarray(t_data)
    y_data = np.asarray(y_data)
    t_data = t_data[start_point:]
    y_data = y_data[start_point:]

    ymax = np.max(y_data)
    init_amp = np.ones(50) * (ymax/50) * 100
    init_decay = np.geomspace(0.1,1e6,50)
    init_log_decay = np.log(init_decay)
    init_offset = [np.mean(y_data[-10:]) * 10] 
    init_guess = np.concatenate([init_amp, init_log_decay,init_offset])

    bounds = ([(0.0,100)] * 50 + [(None,None)] * 50 + [(0.0,10)])


    result = minimize(
        loss_function,
        init_guess,
        args = (t_data,y_data),
        bounds=bounds,
        method = 'L-BFGS-B',
        jac = jacobian_function,
        options={'maxiter': 5000, 'ftol':1e-5}
    )

    fit_amp = result.x[:50] / 100
    fit_decay = np.exp(result.x[50:100]) 
    fit_offset = result.x[100] / 10

    t = 0
    for i in range(0,50):
        t += fit_decay[i] * fit_amp[i]
    print(t)

    #peaks, _ = sps.find_peaks(y_data,distance=max(1,dist), prominence=0.001)
    #if len(peaks) < 3:
    #    return None, "Not enough peaks"
    #t_peaks = t_data[peaks]
    #y_peaks = y_data[peaks]
    #window_size = 3
    #y_summits = np.array([np.max(y_peaks[max(0,i-window_size//2) : min(len(y_peaks), i+window_size//2 + 1)]) for i in range(len(y_peaks))])
#
    #y0_guess = np.mean(y_summits[-max(1,len(y_summits)//5):])
    #a_guess = y_summits[0] - y0_guess
    #t1_guess = (t_peaks[-1]-t_peaks[0]) / 3
    #start_idx = np.argmax(y_summits)
    #tail_floor = np.mean(y_summits[-max(1, len(y_summits)//10):])
    #tail_std = np.std(y_summits[-max(1, len(y_summits)//10):])
    #under_threshold = np.where(y_summits[start_idx:] <= (tail_floor + tail_std))[0]
    #if len(under_threshold) > 0:
    #    end_idx = start_idx + under_threshold[0] + int(len(under_threshold) * 0.2)
    #else:
    #    end_idx = len(y_summits)
    #try:
    #    popt, _ = spo.curve_fit(model_func, t_peaks[start_idx : end_idx],y_summits[start_idx : end_idx], p0=[a_guess,t1_guess,y0_guess], bounds=[0,1e2,0])
    #    fitline = model_func(t_data, *popt)
    #    return popt,fitline, start_idx,end_idx
    #except Exception as e:
    #if mode == "fast":
    #    return Calculate_T_relexation_fast(t_data,y_data,fast_time)
    #if mode == "slow":
    #    return Calculate_T_relexation_slow(t_data,y_data,t_peaks[start_idx],len(t_data))
    #else:
    #    popt, cov = spo.curve_fit(exp_decay, t_peaks[start_idx : end_idx],y_summits[start_idx : end_idx], p0=[a_guess,t1_guess,y0_guess])
    #    fitline = exp_decay(t_data, *popt)
    #    return popt,fitline, start_idx,end_idx,cov
        

def Calculate_T_relexation_slow(t_data, y_data, start, end):
    t = np.array(t_data)
    y = np.array(y_data)
    mask = t < end
    t = t[mask]
    y = y[mask]
    mask = t > start
    t = t[mask]
    y = y[mask]

    def slow(t,A,T,C):
        return A * (1-t/T) + C
    def actual(t,A,T,C):
        return A * np.exp(-t/T) + C
    A = y[0]-y[-1]
    C = y[-1]
    #T = (A-C)/(t[0]-t[-1])
    slope, intercept = np.polyfit(t,y,1)
    T = -A / slope 
    #popt, _ = spo.curve_fit(actual, t,y, p0=[A,T,C])
    fit = actual(t,A,T,C)
    return [A,T,C], fit, 0, end,[[],[0.0,0.0]]
    
def Calculate_T_relexation_fast(t_data, y_data, end):
    def fast(t,A,T, C):
        return A * np.exp(-t/T) + C

    t = np.array(t_data)
    y = np.array(y_data)
    mask = t < end
    C = np.mean(y[-5000:])
    t = t[mask]
    y = y[mask]
    func = partial(fast,C=C)
    try:
        popt, cov  = spo.curve_fit(func, t , y, p0 = [0.6,1000])
        fitline = func(t, *popt)
        return popt,fitline, 0,end,cov
    except Exception as e:
        return None, f"Fit failed: {str(e)}"

def determine_fit_procedure(t_data, y_data, slow_threshold = 0.05, fast_threshold = 0.8, normal_threshold = 0.5, normal_slope = 10, hl_threshold = 0.02):
    total_range = y_data[0] - y_data[-1]
    fluc = np.max(y_data) - np.min(y_data)
    
    tp = int(len(y_data) * 0.1)
    start = y_data[:tp]
    mid = y_data[tp:-tp]
    end = y_data[-tp:]
    total_drop = np.mean(start) - np.mean(end)
    noise = np.std(end)


    if total_range < slow_threshold:
        return "slow"
    if fluc < (3*noise):
        return "slow"

    drop_to_mid = np.mean(start) - np.mean(mid)
    if drop_to_mid > (fast_threshold * total_drop) and drop_to_mid < total_drop:
        return "fast"
    elif drop_to_mid > (normal_threshold * total_drop) and drop_to_mid < total_drop:
        return "normal"
    elif drop_to_mid < total_drop:
        return "slow"

    slope_1 = np.polyfit(t_data[:tp], y_data[:tp], 1)[0]
    slope_2 = np.polyfit(t_data[-1*tp:], y_data[-1*tp:], 1)[0]
    slope_ratio = abs(slope_1 / slope_2) if slope_2 != 0 else 1000
    if slope_ratio > normal_slope:
        half_life_idx = np.where(y_data < (y_data[0] - total_range/2))[0]
        if len(half_life_idx) > 0 and half_life_idx[0] < len(y_data) * hl_threshold:
            return "fast"
        return "normal" 
        
    return "slow"
    
    #slope, _ = np.polyfit(t_data,y_data,1)
    #inital_drop = y_data[0] - y_data[int(len(y_data)*0.1)]
    #if total_range < (3*noise_level):
    #    return "slow"
    #if inital_drop >= (1-fast_threshold)*total_range and abs(slope) > fast_slope:
    #    return "fast"
    #if abs(slope) < slow_threshold:
    #    return "slow"
    #return "normal"

def get_time_index(t_data,time):
    idx = np.searchsorted(t_data, time)
    if idx < len(t_data):
        return idx
    else:
        return len(t_data)

        
def PeakEnvelopeFit(y_data, t_data):
    y_data_d = sps.detrend(y_data)
    y = y_data_d - np.mean(y_data_d)
    x = t_data

    if CUDA:
        freq, power = ats.LombScargle(x,y,nterms=1).autopower(method="fastnifty")
    else:
        freq, power = ats.LombScargle(x,y,nterms=1).autopower()
    best_freq = freq[np.argmax(power)]
    dominant_period = 1 / best_freq
    avg_peak_distance = dominant_period
    #
    peaks, _ = sps.find_peaks(y_data, distance=int(avg_peak_distance * 2))
    peaks = peaks
    t_peaks = t_data[peaks]
    y_peaks = y_data[peaks]

    peaks2, _  = sps.find_peaks(y_peaks, distance = 1, prominence = 0.01)
    t_peaks_2 = t_peaks[peaks2]
    y_peaks_2 = y_peaks[peaks2]

    windows_size = 4
    y_summits= spd.maximum_filter1d(y_peaks_2,size=windows_size)

    envelope_func = spint.interp1d(t_peaks, y_peaks, kind='cubic', fill_value='extrapolate')
    upper_env = envelope_func(t_data)

    def model_curve(t, A, T, C):
        return A * np.exp(-t/T) + C
    

    intial_guess = [y_summits[0], 1000, 0]
    popt_top, _ = spo.curve_fit(model_curve, t_peaks_2, y_summits, p0=intial_guess)
    fit_line_top = model_curve(t_data, *popt_top)
    A_fit, T_fit, C_fit = popt_top


    fig,ax = plt.subplots()
    ax.plot(t_data, y_data, alpha=0.4, label='Data')
    ax.plot(t_data, upper_env, 'r', linewidth=2, label='Peak Envelope')
    ax.plot(t_peaks_2, y_peaks_2, 'kx', label='Peaks')
    #ax.plot(t_data, fit, 'g--', linewidth=2, label=f'Fit: T={T_fit:.2f} ns')
    ax.plot(t_data, fit_line_top, 'b--', linewidth=2, label=f'Fit: T={T_fit:.2f} ns')
    plt.xlabel('Time (ns)')
    plt.ylabel('Standard Deviation')
    plt.title('Peak Envelope Fit')
    plt.legend()
    plt.show()

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
    std = []
    for i in range(5,len(filenames)):
        t, m, sd, sdsd = plot_standard_deviation(dat,'gs.t0', i,window)
        time_idx = get_time_index(t,warm_up)
        #mode_str = determine_fit_procedure(t[time_idx:],m[time_idx:])
        mode_str = regime[i]
        print("MODE: " + mode_str)

        bound_T = partial(Calculate_T_relaxation,t,sdsd,1000, dist = 10000, mode=mode_str, fast_time =10000)
        indicies = dat._index_map[dat.unique_groups[i]]
        time = dat.data['Time(ns)'][indicies]
        signal = dat.data['gs.t0'][indicies]
        strain = dat.data['gs.strain.ex'][indicies][0]
        T, _,C,S = Plot_T_on_TimeEvo(time,signal, bound_T)
        T_times.append(T)
        C_val.append(C)
        x.append(strain)
        std.append(S)
        #plt.close()
    
    fig,ax = plt.subplots()
    ax.errorbar(x,T_times,yerr=std, fmt='o', capsize=5, capthick=1, color='blue',ecolor='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
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

    

    #peak_


    t,sol = rate_kinetics(kinetics)

    #plot_kinetics(states[0] + states[1] + states[2])


if __name__ == "__main__":
    #main()
    main(True)