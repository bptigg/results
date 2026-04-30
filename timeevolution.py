
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
    
    ewma = dm.ExponentiallyWeightedMovingAverage(window)
    std_devs = []
    std_devs_sig = []
    means = []
    means_sig = []
    for i in range(1,num_steps):
        dt = t[i]-t[i-1]
        val = y[i]
        sigma,mean,sigma_sig, mean_sig = ewma.update(val,dt)
        std_devs.append(sigma)
        means.append(mean)
        means_sig.append(mean_sig)
        std_devs_sig.append(sigma_sig)

    
    fig,ax = plt.subplots()
    t_plot = t[1:num_steps]
    ax.plot(t_plot,std_devs, label = rf"$\sigma_{{{state}}}$")
    ax.plot(t_plot,means, label = rf"$\langle{{{state}}}\rangle$")
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('State population')
    ax.legend()
    #ax.vlines([window,2*window,3*window,4*window,5*window],[0,0,0,0,0],[1,1,1,1,1])
    ax.axvspan(0,5*window, color = (117/255,124/255,136/255,0.5))
    plt.draw()

    fig2,ax2 = plt.subplots()
    t_plot = t[1:num_steps]
    ax2.plot(t_plot,std_devs_sig, label = rf"$\sigma_{{\sigma_{{{state}}}}}$")
    #ax2.plot(t_plot,means_sig)
    #ax.vlines([window,2*window,3*window,4*window,5*window],[0,0,0,0,0],[1,1,1,1,1])
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('State population')
    ax2.legend()
    ax2.axvspan(0,100*window, color = (117/255,124/255,136/255,0.5))
    plt.show(block=False)
    return t_plot, mean, std_devs, std_devs_sig
    
    popt, line, start, end = Calculate_T_relaxation(t_data=t_plot, y_data=std_devs_sig, start_point=100*window)
    T = popt[1]
    y0_pop = np.mean(y[-int(len(y)*0.1):])
    A_pop = y[0] - y0_pop
    pop_fit = A_pop * np.exp(-(t-t[0]) / T) + y0_pop
    
    fig3, ax3 = plt.subplots()
    ax3.plot(t,pop_fit)
    plt.show()
    #std_devs[0] = np.float64(std_devs[0])
    #std_devs = np.array(std_devs)
    #PeakEnvelopeFit(std_devs,t_plot)

def Plot_T_on_TimeEvo(t_data, y_data, T_Calc_func : function):
    popt, line, start, end = T_Calc_func()
    t = np.array(t_data)
    y = np.array(y_data)
    T = popt[1]
    C = np.mean(y[-int(len(y)*0.1):])
    A = y[0]- C
    fit_func = A * np.exp(-(t-t[0]) / T) + C
    fig, ax = plt.subplots()
    ax.plot(t,y)
    ax.plot(t,fit_func,'r--')
    plt.draw()

def Calculate_T_relaxation(t_data, y_data, start_point, dist = 5000):
    t_data = np.asarray(t_data)
    y_data = np.asarray(y_data)
    t_data = t_data[start_point:]
    y_data = y_data[start_point:]

    peaks, _ = sps.find_peaks(y_data,distance=max(1,dist), prominence=0.001)
    if len(peaks) < 3:
        return None, "Not enough peaks"
    t_peaks = t_data[peaks]
    y_peaks = y_data[peaks]
    window_size = 3
    y_summits = np.array([np.max(y_peaks[max(0,i-window_size//2) : min(len(y_peaks), i+window_size//2 + 1)]) for i in range(len(y_peaks))])
    def model_func(t, A, T1, y0):
        return A * np.exp(-t/T1) + y0
    y0_guess = np.mean(y_summits[-max(1,len(y_summits)//5):])
    a_guess = y_summits[0] - y0_guess
    t1_guess = (t_peaks[-1]-t_peaks[0]) / 3
    start_idx = np.argmax(y_summits)
    tail_floor = np.mean(y_summits[-max(1, len(y_summits)//10):])
    tail_std = np.std(y_summits[-max(1, len(y_summits)//10):])
    under_threshold = np.where(y_summits[start_idx:] <= (tail_floor + tail_std))[0]
    if len(under_threshold) > 0:
        end_idx = start_idx + under_threshold[0] + int(len(under_threshold) * 0.2)
    else:
        end_idx = len(y_summits)
    try:
        popt, _ = spo.curve_fit(model_func, t_peaks[start_idx : end_idx],y_summits[start_idx : end_idx], p0=[a_guess,t1_guess,y0_guess])
        fitline = model_func(t_data, *popt)
        return popt,fitline, start_idx,end_idx
    except Exception as e:
        return Calculate_T_relexation_fast(t_data,y_data,25000)
    
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
        popt, _  = spo.curve_fit(func, t , y, p0 = [0.6,1000])
        fitline = func(t, *popt)
        return popt,fitline, 0,end
    except Exception as e:
        return None, f"Fit failed: {str(e)}"
        
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
    extension = "2"
    npzfile = ""
    #file = file_dict + file_name# + extension
    file = file_name + extension
    if(not load):
        npzfile = dm.ProcessFile(file,file_name+extension)
        #npzfile = dm.ProcessFileSSH(file_name, file_name,file_dict_ssh,"scandium.qbl.uni-oldenburg.de","juft2450")
    else:
        npzfile = file + ".npz"
    #dat = DataPointCollection(npzfile)
    dat = DataPointCollection(file_name + extension + ".npz")

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
    t, m, sd, sdsd = plot_standard_deviation(dat,'gs.t0', 0,100)
    bound_T = partial(Calculate_T_relaxation,t,sdsd,1000, dist = 10000)
    indicies = dat._index_map[dat.unique_groups[0]]
    time = dat.data['Time(ns)'][indicies]
    signal = dat.data['gs.t0'][indicies]
    Plot_T_on_TimeEvo(time,signal, bound_T)
    plt.show()

    

    #peak_


    t,sol = rate_kinetics(kinetics)

    #plot_kinetics(states[0] + states[1] + states[2])


if __name__ == "__main__":
    #main()
    main(True)