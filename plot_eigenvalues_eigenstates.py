#plot eigenvalues with eigenstates

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
import math
import mmap

class DataPoint:
    def __init__(self, x, eigenvalues, state_index, weighting,colour, state):
        self.eigenvalues = eigenvalues
        self.weighting = weighting
        self.state_index = state_index
        self.x = x
        self.colour = colour
        self.state = state

class DataPointCollection:
    def __init__(self,filename):
        self.data = np.load(filename, allow_pickle=True, mmap_mode='r')
        self._group_map = self.data['_group_map']
        self._time_steps = self.data['_timesteps']
        self.unique_groups = np.unique(self._group_map)
        self.cached_keys = [k for k in self.data.files if k != '_group_map']

    def __len__(self):
        return len(self.unique_groups)
    def __getitem__(self, key):
        indices = np.where(self._group_map == key)[0]
        return [DataView(self.data, self.GetKeys(), i) for i in indices]
    def GetKeys(self):
        return self.cached_keys

class DataView:
    def __init__(self, master_collection,keys, index):
        self.master = master_collection
        self.i = index
        self.keys = keys
        self._cache = {}

    def __getattr__(self, name):
        if name in self._cache:
            return self._cache[name]
        
        if name in self.keys:
            val = self.master[name][self.i]
            if hasattr(val,'item') and not isinstance(val, np.ndarray):
                val = val.item()
            self._cache[name] = val
            return val
        raise AttributeError(f"'DataView' object has no attribute '{name}'")
    
    def __repr__(self):
        return f"<DataView index={self.i} header={self.Header}>"

def load_block(collection : DataPointCollection, idx):
    mask = collection._group_map == collection.unique_groups[idx]
    dat_x = collection.data['x'][mask]
    #dat_y = collection.data['sc'][mask]
    dat_c = collection.data['colour'][mask]
    dat_z = collection.data['eigenvalues'][mask]

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
def StartColourMap(eigenvectors):
    global cmp_init
    if cmp_init:
        return
    #cmap = mpl.colormaps['magma']
    #colors = cmap(np.linspace(0, 1, eigenvectors))
    cmap = mpl.colormaps['tab10']
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

def CollectEigenstatesTR(data, eigenvalues, offset, x_index, y_index = -1, sets : int = 1, state_headers = [], max_steps = -1):
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
                    DataPoints[step][d-offset_D].append(DataPoint([x],[eigenvalue_value[set_idx][eig]], eig, dat[2][state_id][1],[(0.0, 0.0, 0.0, 1.0)], ['mixed']))
        idx += 1
    return DataPoints



def GetStateHeader(lineheader, eigenvalues, ignore_indices, offset):
    state_headers = []
    for i in range(0,eigenvalues):
        if i in ignore_indices:
            continue
        state_headers.append(lineheader[offset + ((i+1) * eigenvalues)])
    return state_headers

def assign_state_to_colour(data_points, state_headers, sets):
    state_to_colour = {}
    all_states = False
    num_states_found = 0
    num_states = len(data_points)
    state_to_colour['mixed'] = (0.0,0.0,0.0,1.0)
    for DP in data_points: 
        for dp in DP:
            #if dp.eigenvalues == []:
            #    continue
            state = dp.state[0]
            colour = dp.colour[0]
            if state != 'mixed' and state not in state_to_colour:
                state_to_colour[state] = colour
                num_states_found += 1
            if(num_states_found == num_states):
                all_states = True
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
    xs = [o for d in DataPoints for i in d for o in i.x]
    ys = [o for d in DataPoints for i in d for o in i.eigenvalues]
    cs = [o for d in DataPoints for i in d for o in i.colour]
    state_dict = assign_state_to_colour(DataPoints, lineheader)
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
        for s in range(sets):
            fig, ax = plt.subplots()
            xs, ys, cs = load_block(DataPoints,p)
            xs = [o for x in range(0,len(xs),sets*eig) for o in xs[x + (s*eig):x + ((s+1)*eig)]]
            ys = [o for x in range(0,len(ys),sets*eig) for o in ys[x + (s*eig):x + ((s+1)*eig)]]
            cs = [o for x in range(0,len(cs),sets*eig) for o in cs[x + (s*eig):x + ((s+1)*eig)]]
            modified_data_points = DataPoints[p][0:sets*eig][s*eig:(s+1)*eig]
            state_dict = assign_state_to_colour([modified_data_points], lineheader[s], sets=sets)
            ax.scatter(xs, ys, color=cs, s=20, edgecolors='none')

            legend_elements = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=10, label=legend_label) for legend_label, color in state_dict.items()
            ]
            ax.legend(handles=legend_elements, title="Eigenstates", loc='upper right')

            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Eigenvalues (rad/ns)')
            plt.show()

def save(filename, eigenvalues):
    twod_list = []
    group_ids = []
    timestep = []
    for group_idx, sub_list in enumerate(eigenvalues):
        for p in sub_list:
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

    np.savez_compressed(filename, **save_dict)
    print(f"Saved variables: {list(sample_vars)}")

def load(filename):
    filename = filename + ".npz"
    return DataPointCollection(filename)



def main():
    lines = read_file("NV_centre_N14_0T_eig.dat")
    eigenvalues = 9
    offset = 3
    Header = lines[0]
    #state_headers = GetStateHeader(Header, eigenvalues, [6,7,8])
    state_headers = GetStateHeader(Header, eigenvalues, [])
    Data = lines[1:len(lines)]
    DataPoints = CollectEigenstates(Data, eigenvalues, offset, 2, state_headers)
    PlotEigenstates(DataPoints,state_headers)

def PlotTimeResolvedEigenStates():
    #lines = read_file("NV_centre_N14_01T_eig-test.dat")
    lines = fast_read("NV_centre_N14_01T_eig-test.dat")
    eigenvalues = 9
    offset = 4
    Header = lines[0]
    #state_headers = GetStateHeader(Header, eigenvalues, [])
    state_headers = [["GS.T0_U", "GS.T0_Z", "GS.T0_D", "GS.TP_U", "GS.TP_Z", "GS.TP_D", "GS.TD_U", "GS.TD_Z", "GS.TD_D"], ["ES.T0_U", "ES.T0_Z", "ES.T0_D", "ES.TP_U", "ES.TP_Z", "ES.TP_D", "ES.TD_U", "ES.TD_Z", "ES.TD_D"]]
    Data = lines[1:len(lines)]
    DataPoints = CollectEigenstatesTR(Data, eigenvalues, 4, 1, sets=2, state_headers=state_headers, max_steps=5)
    save("NV_centre_N14_01T_eig-test", DataPoints)
    DataPoints = load("NV_centre_N14_01T_eig-test")
    PlotTREigenStates(DataPoints,state_headers, sets=2,eig=eigenvalues)





#main()
PlotTimeResolvedEigenStates()


