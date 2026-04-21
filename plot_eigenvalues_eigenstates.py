#plot eigenvalues with eigenstates

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
import math
import mmap
from DataManagment import DataPointCollection, DataView

class DataPoint:
    def __init__(self, x, eigenvalues, state_index, weighting,colour, state):
        self.eigenvalues = eigenvalues
        self.weighting = weighting
        self.state_index = state_index
        self.x = x
        #self.y = y
        self.colour = colour
        self.state = state


def load_block(collection : DataPointCollection, idx):
    collection.last_accessed_group = idx
    group_id = collection.unique_groups[idx]
    indicies = collection._index_map[group_id]
    dat_x = collection.data['x'][indicies]
    #dat_y = collection.data['sc'][mask]
    dat_c = collection.data['colour'][indicies]
    dat_z = collection.data['eigenvalues'][indicies]

    return dat_x, dat_z, dat_c


def read_file(filename):
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            seperator = ' '
            strs = []
            current_string = ""
            for c in line:
                if c == seperator:
                    strs.append(current_string)
                    current_string = ""
                else:
                    current_string += c
            #print(strs)
            lines.append(strs)
    return lines

def fast_read(filename):
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            lines = []
            for line in iter(mm.readline, b""):
                lines.append(line.decode('utf-8').split())
            return lines

def GetMax(values):
    threshold = 0.95
    min_threshold = 0.05
    max_value = 0
    max_index = 0
    index = 0
    valid_states = []
    for v in values:
        if v > min_threshold:
            valid_states.append(v)
        else:
            valid_states.append(0)
        if v > max_value:
            max_value = v
            max_index = index
        if v == 1:
            return index,valid_states, False
        index += 1

    over_min = 0
    for i in valid_states:
        if(i > min_threshold):
            over_min += 1

    if(valid_states[max_index] < threshold or over_min > 1):
        return max_index,valid_states, True
    return max_index,valid_states, False

cmp_init = False
colours_dict ={}
def StartColourMap(eigenvectors, map = 'tab10'):
    global cmp_init
    if cmp_init:
        return
    #cmap = mpl.colormaps['magma']
    #colors = cmap(np.linspace(0, 1, eigenvectors))
    cmap = mpl.colormaps[map]
    colors = cmap.colors
    cmp_init = True
    for i, colour in enumerate(colors):
        colours_dict[i] = (colour[0], colour[1], colour[2], 1.0)

def PlotEigenvaluesWithWeightings(B : list, eigenvalues : list, weightings : list):
    #2d plot of the eigenvalues with the weightings as different colours for each eigenvector
    for i in range(len(weightings[0])):
        fig, ax = plt.subplots()
        y = []
        cs = []
        xs = []
        for j in range(len(eigenvalues)):
            for k in range(len(eigenvalues[j])):
                xs.append(B[j])
                y.append(eigenvalues[j][k])
                cs.append(weightings[j][i][k])
        sc = ax.scatter(xs, y, c=cs, cmap='viridis', s=50)
        plt.colorbar(sc, label='Weighting')
        ax.set_xlabel('Magnetic Field (T)')
        ax.set_ylabel('Eigenvalues (meV)')
        plt.show()

def CollectEigenstates(data, eigenvalues, offset, x_index, state_headers):
    ##B = []
    E = []
    W = []
    DataPoints = []
    global colours_dict
    B_all = []
    StartColourMap(eigenvalues+1)
    for d in range(0,len(data)):
        B = float(data[d][x_index])
        DataPoints.append([])
        weightings_list = []
        B_all.append(B)
        for i in range(0,eigenvalues):
            weightings = []
            eigenvalue = []
            for eig in range(0,eigenvalues):
                eigenvalue_value = float(data[d][offset + eig])
                #if(abs(eigenvalue_value) > 20):
                #    continue
                eigenvalue.append(eigenvalue_value)
                state_index = offset + eigenvalues + (eig * eigenvalues) + i
                weightings.append(float(data[d][state_index]))
            #StartColourMap(len(eigenvalue)) 
            max_index,v,mixed = GetMax(weightings)
            if mixed == True:
                index = [x for x in range(len(v)) if v[x] > 0]
                eig_values = [eigenvalue[x] for x in range(len(v)) if v[x] > 0]
                if(eig_values == []):
                    eig_values = [eigenvalue[x] for x in range(len(weightings)) if weightings[x] > 1e-6]
                #    index = [x for x in range(len(v)) if weightings[x] > 1e-6]
                #mixed_color = []
                #for i in index:
                #    mixed_color[0] += colours_dict[i]
                DataPoints[d].append(DataPoint([B for x in eig_values], eig_values, max_index, weightings[max_index], [(0.0,0.0,0.0,1.0) for x in eig_values], ['mixed' for x in eig_values]))
                #DataPoints[d].append(DataPoint(B, eigenvalue[max_index], max_index, weightings[max_index], colours_dict[i], state_headers[i]))
            else:
                DataPoints[d].append(DataPoint([B], [eigenvalue[max_index]], max_index, weightings[max_index], [colours_dict[i]], [state_headers[i]]))
            weightings_list.append(weightings)
        E.append(eigenvalue)
        W.append(weightings_list)
    #PlotEigenvaluesWithWeightings(B_all, E, W)
    return DataPoints

def CollectEigenstatesTR(data, eigenvalues, offset, x_index, y_index = -1, sets : int = 1, state_headers = [], max_steps = -1, max_timestep = -1):
    DataPoints = [[]]
    global colours_dict
    StartColourMap(eigenvalues+1)
    step = int(data[0][0]) - 1
    old_step = step
    idx = 0
    offset_D = 0
    for d in range(0,len(data)):
        step = int(data[d][0]) - 1
        if(step != old_step):
            DataPoints.append([])
            old_step = step
            idx = 0
            offset_D = d
            if(max_steps != -1):
                if(step >= max_steps):
                    DataPoints.remove([])
                    break
        if(max_timestep != -1):
            if(float(data[d][1]) >= max_timestep):
                continue
        DataPoints[step].append([])
        x = float(data[d][x_index])
        weights = [[] for _ in range(sets)]
        eigenvalue_value = [[] for _ in range(sets)]
        index = [[] for _ in range(sets)]
        y = 0
        if(y_index != -1):
            y = float(data[d][y_index])
        for eig in range(0,eigenvalues * sets):
            set_idx = math.floor(eig/eigenvalues)
            eig_idx = eig % eigenvalues
            col = offset + ((eigenvalues + 1) * eigenvalues * set_idx) + eig_idx
            eigenvalue_value[set_idx].append(float(data[d][col]))
            weights[set_idx].append([float(data[d][col + (i+1)*eigenvalues]) for i in range(eigenvalues)])
            index[set_idx].append([set_idx,eig_idx,[]])

        set_idx = 0
        for w in weights:
            for i in range(len(state_headers[set_idx])):
                max_index,v,mixed = GetMax(w[i])
                if(not mixed):
                    index[set_idx][max_index][2].append([i,1])
                if mixed:
                    indicies = [x for x in range(len(v)) if v[x] > 0]
                    for idx in indicies:
                        index[set_idx][idx][2].append([i,v[idx]])
            set_idx += 1

        for set_idx in range(sets):
            for eig in range(eigenvalues):
                dat = index[set_idx][eig]
                if (len(dat[2]) == 1):
                    state = dat[2][0][0]
                    DataPoints[step][d-offset_D].append(DataPoint([x],[eigenvalue_value[set_idx][eig]], eig, dat[2][0][1],[colours_dict[state]], [state_headers[set_idx][state]]))
                else:
                    id = 0
                    state_id = 0
                    for i in range(0,len(dat[2])):
                        if dat[2][i][1] > id:
                            id = dat[2][i][1]
                            state_id = i
                    #state = dat[2][state_id][0]
                    #total_color = colours_dict[state]
                    #sum all the colors together weighted by their contribution to the state
                    #total_color = (0.0, 0.0, 0.0, 1.0)
                    #for i in range(0,len(dat[2])):
                    #    state = dat[2][i][0]
                    #    contribution = dat[2][i][1]
                    #    color = colours_dict[state]
                    #    total_color = (total_color[0] + color[0] * contribution, total_color[1] + color[1] * contribution, total_color[2] + color[2] * contribution, 1.0)
                    #DataPoints[step][d-offset_D].append(DataPoint([x],[eigenvalue_value[set_idx][eig]], eig, dat[2][state_id][1], [total_color], ['mixed']))
                    #need to find a better way to represent the mixed states that doesn't just average the colors but asssignes a new color
                    color_contributions = [colours_dict[eig]]
                    states = []
                    for i in range(0,len(dat[2])):
                        state = dat[2][i][0]
                        contribution = dat[2][i][1]
                        color = colours_dict[state]
                        color_contributions.append((color[0], color[1], color[2], contribution))
                        states.append('line ' + str(eig))
                    DataPoints[step][d-offset_D].append(DataPoint([x],[eigenvalue_value[set_idx][eig]], eig, dat[2][state_id][1], color_contributions, states))
        idx += 1
    return DataPoints



def GetStateHeader(lineheader, eigenvalues, ignore_indices, offset):
    state_headers = []
    for i in range(0,eigenvalues):
        if i in ignore_indices:
            continue
        state_headers.append(lineheader[offset + ((i+1) * eigenvalues)])
    return state_headers

def assign_state_to_colour(data_points, state_headers, sets = 2, max_states = -1):
    state_to_colour = {}
    all_states = False
    num_states_found = 0
    if(max_states != -1):
        num_states = max_states
    else:
        num_states = len(data_points)
    state_to_colour['mixed'] = (0.0,0.0,0.0,1.0)
    for DP in data_points: 
        index = 0
        for dp in DP:
            #if dp.eigenvalues == []:
            #    continue
            if(state_headers != []):
                state = state_headers[index]
            else:
                state = dp.state[0]
            colour = dp.colour[0]
            if state != 'mixed' and state not in state_to_colour:
                state_to_colour[state] = colour
                num_states_found += 1
            if(num_states_found == num_states):
                all_states = True
            index += 1
        if(all_states):
            break
    return state_to_colour

def PlotEigenstates(DataPoints, lineheader):
    fig, ax = plt.subplots()
    #segments= [np.column_stack((o.x,o.eigenvalues)) for d in DataPoints for o in d]
    #colors = [o.colour for d in DataPoints for o in d]
    #lc = LineCollection(segments, colors=colors, linewidths=2, alpha=0.7)
    #ax.add_collection(lc)
    #ax.set_xlim(ax.dataLim.x0, ax.dataLim.x1)
    #ax.set_ylim(ax.dataLim.y0, ax.dataLim.y1)
    xs,ys,cs = [],[],[]
    all_indicies = np.concatenate([DataPoints._index_map[group_id] for group_id in DataPoints.unique_groups])

    def flatten(lst):
        if lst.dtype != object:
            return lst.ravel()
        return np.concatenate(lst)
    
    xs = flatten(DataPoints.data['x'][all_indicies])
    ys = flatten(DataPoints.data['eigenvalues'][all_indicies])
    cs = flatten(DataPoints.data['colour'][all_indicies])
    #for d in range(0,len(DataPoints)):
    #    x,y,c = load_block(DataPoints,d)
    #    xs.append(x.ravel())
    #    ys.append(y.ravel())
    #    cs.append(c.ravel())
    #    #x2 = [o for i in x for o in i]
    #    #y2 = [o for i in y for o in i]
    #    #c2 = [o for i in c for o in i]
    #    #xs = np.concatenate((xs,x2))
    #    #ys = np.concatenate((ys,y2))
    #    #if len(cs) == 0:
    #    #    cs = c2
    #    #else:
    #    #    cs = np.concatenate((cs,c2))
    #xs = [o for d in DataPoints for i in d for o in i.x]
    #ys = [o for d in DataPoints for i in d for o in i.eigenvalues]
    #cs = [o for d in DataPoints for i in d for o in i.colour]
    state_dict = assign_state_to_colour(DataPoints, [], sets=1, max_states=len(lineheader))
    ax.scatter(xs, ys, color=cs, s=20, edgecolors='none')

    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=10, label=legend_label) for legend_label, color in state_dict.items()
    ]
    ax.legend(handles=legend_elements, title="Eigenstates", loc='upper right')

    #for d in range(0,len(DataPoints)):
    #    for i in range(0,len(DataPoints[d])):
    #        dp = DataPoints[d][i]
    #        ax.scatter(dp.x, dp.eigenvalues, s=50*dp.weighting, color=dp.colour, alpha=0.7, edgecolors='none')
    ax.set_xlabel('Magnetic Field (T)')
    ax.set_ylabel('Eigenvalues (meV)')
    plt.show()

def PlotTREigenStates(DataPoints, lineheader, sets,eig):
    plots = len(DataPoints)
    for p in range(plots):
        xs, ys, cs = load_block(DataPoints,p)
        x_grid = xs.reshape(-1, sets, eig)
        y_grid = ys.reshape(-1, sets, eig)
        c_grid = cs.reshape(len(x_grid), sets, eig, -1)
        #num_blocks = xs.shape[0] // (sets*eig)
        for s in range(sets):
            x_dat = x_grid[:,s,:].ravel()
            y_dat = y_grid[:,s,:].ravel()
            c_dat = c_grid[:,s,:,:].reshape(-1, c_grid.shape[-1])
            #x = [o for x in range(0,len(xs),sets*eig) for o in xs[x + (s*eig):x + ((s+1)*eig)]]
            #y = [o for x in range(0,len(ys),sets*eig) for o in ys[x + (s*eig):x + ((s+1)*eig)]]
            #cs = [o for x in range(0,len(cs),sets*eig) for o in cs[x + (s*eig):x + ((s+1)*eig)]]
            #c =[o[0] for o in cs]
            #c = [(o[0], o[1], o[2], 1.0) for o in c]
            #c_flat = cs.reshape(-1, cs.shape[-1])
            #c = np.column_stack((c_flat[:,0], c_flat[:,1], c_flat[:,2], np.ones(c_flat.shape[0])))

            #modified_data_points = DataPoints[p][0:sets*eig][s*eig:(s+1)*eig]
            headers = [f"line {i}" for i in range(eig)]
            #state_dict = assign_state_to_colour([modified_data_points], headers, sets=sets)
            #PlotMixedStateContributions(x,cs, lineheader[s],eig, state_dict, headers)
            fig, ax = plt.subplots()
            #ax.scatter(x, y, color=c, s=20, edgecolors='none')
            ax.scatter(x_dat, y_dat, s=20, edgecolors='none')

            #legend_elements = [
            #Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=10, label=legend_label) for legend_label, color in state_dict.items()
            #]
            #ax.legend(handles=legend_elements, title="Eigenstates", loc='upper right')

            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Eigenvalues (rad/ns)')
            print(f"Plotted set {s} of plot {p}")
            #plt.show(block=False)
            #input("Close plots")
            #plt.close('all')
            #plt.draw()
            plt.show()

def PlotMixedStateContributions(x, contributions, lineheader,eig, state_dict, headers):
    y_all = []
    c_all = []
    for i in range(0,len(contributions),eig):
        y_t = []
        c_t = []
        for e in range(eig):
            contribution = contributions[i + e]
            y = []
            c_list = []
            for c in contribution[1:]:
                color = (c[0], c[1], c[2], 1.0)
                value = c[3]
                y.append(value)
                c_list.append(color)
            y_t.append(y)
            c_t.append(c_list)
        y_all.append(y_t)
        c_all.append(c_t)
    for i in range(eig):
        fig, ax = plt.subplots()
        y = [y_all[t][i] for t in range(len(y_all))]
        c = [c_all[t][i] for t in range(len(c_all))]
        y_unpacked = []
        c_unpacked = []
        time = []
        for t in range(len(y)):
            for s in range(len(y[t])):
                y_unpacked.append(y[t][s])
                c_unpacked.append(c[t][s])
                time.append(x[t*eig])
        ax.scatter(time, y_unpacked, color=c_unpacked, s=20, edgecolors='none')
        all_colors = list(set(tuple(color) for sublist in c for color in sublist))
        states = [lineheader[StateByColor(color)] for color in all_colors]

        legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=all_colors[i], markersize=10, label=states[i]) for i in range(len(all_colors))
        ]
        ax.legend(handles=legend_elements, title="Eigenstates", loc='upper right')

        ax.set_title(f"Overlap of reference states to the mixed state in {headers[i]}")
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Overlap')
        plt.draw()
    plt.show()
        

def StateByColor(color):
    if color in colours_dict.values():
        for state, col in colours_dict.items():
            if col == color:
                return state
    return None


def save(filename, eigenvalues):
    twod_list = []
    group_ids = []
    timestep = []
    for group_idx, sub_list in enumerate(eigenvalues):
        for p in sub_list:
            if type(p) != list:
                p = [p]
            for i in range(len(p)):
                group_ids.append(group_idx)
            twod_list.append(p)
    flat_list = []
    for ts, sub_list in enumerate(twod_list):
        for p in sub_list:
            timestep.append(ts)
            flat_list.append(p)

    sample_vars = flat_list[0].__dict__
    save_dict = {'_group_map': group_ids, '_timesteps' : timestep}
    #save_dict = {'_timesteps' : timestep}

    for var_name in sample_vars:
        data = [getattr(p,var_name) for p in flat_list]
        save_dict[var_name] = np.array(data, dtype=object)

    np.savez(filename, **save_dict)
    print(f"Saved variables: {list(sample_vars)}")

#def GetEnergyDifference(data, state1, state2):
    #energy_diff = []

def load(filename):
    filename = filename + ".npz"
    return DataPointCollection(filename)



def main():
    #lines = read_file("NV_centre_N14_0T_eig.dat")
    file = "NV_centre_N14_GLASC_Z_eig_strain"
    lines = fast_read(file + ".dat")
    eigenvalues = 9
    offset = 3 #for gs
    #for es 
    offset_es = 3 + (eigenvalues + 1) * eigenvalues
    #Header = lines[0]
    #state_headers = GetStateHeader(Header, eigenvalues, [6,7,8])
    #state_headers = GetStateHeader(Header, eigenvalues, [], offset)
    state_headers = [["GS.T0_U", "GS.T0_Z", "GS.T0_D", "GS.TD_U", "GS.TD_Z", "GS.TD_D", "GS.TP_U", "GS.TP_Z", "GS.TP_D"], ["ES.T0_U", "ES.T0_Z", "ES.T0_D", "ES.TP_U", "ES.TP_Z", "ES.TP_D", "ES.TD_U", "ES.TD_Z", "ES.TD_D"]]
    Data = lines[1:len(lines)]
    #DataPoints = CollectEigenstates(Data, eigenvalues, offset_es, 2, state_headers[1])
    DataPoints = CollectEigenstates(Data, eigenvalues, offset, 2, state_headers[0])
    #save(file + "_es", DataPoints)
    save(file + "_gs", DataPoints)
    StartColourMap(eigenvalues+1)
    #DataPoints = load(file + "_es")
    DataPoints = load(file + "_gs")
    PlotEigenstates(DataPoints,state_headers[0])
    #plt.show()

def PlotTimeResolvedEigenStates():
    #lines = read_file("NV_centre_N14_01T_eig-test.dat")
    #file = "NV_centre_N14_01T_eig-test"
    file = "NV_centre_N14_GLASC_Z_eig_strain_mod"
    #file = "eigenvalues_gs_Z"
    #lines = fast_read(file + ".dat")
    eigenvalues = 9
    offset = 3
    #Header = lines[0]
    #state_headers = GetStateHeader(Header, eigenvalues, [])
    state_headers = [["GS.T0_U", "GS.T0_Z", "GS.T0_D", "GS.TP_U", "GS.TP_Z", "GS.TP_D", "GS.TD_U", "GS.TD_Z", "GS.TD_D"], ["ES.T0_U", "ES.T0_Z", "ES.T0_D", "ES.TP_U", "ES.TP_Z", "ES.TP_D", "ES.TD_U", "ES.TD_Z", "ES.TD_D"]]
    #Data = lines[1:len(lines)]
    #DataPoints = CollectEigenstatesTR(Data, eigenvalues, offset, 1, sets=2, state_headers=state_headers, max_timestep = 100)
    #DataPoints = CollectEigenstates(Data, eigenvalues, offset, 2, state_headers[0])
    #save(file, DataPoints)
    StartColourMap(eigenvalues+1)
    DataPoints = load(file)
    PlotTREigenStates(DataPoints,state_headers, sets=2,eig=eigenvalues)
    #PlotEigenstates(DataPoints,state_headers[0])
    #GetEnergyDifference(DataPoints, "GS.T0_U", "GS
    plt.show()





#main()
#PlotTimeResolvedEigenStates()
if __name__ == "__main__":
    #main()
    PlotTimeResolvedEigenStates()


