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

#class DataPointCollection

#class DataView


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
    for d in range(0,len(data)):
        step = int(data[d][0]) - 1
        if(step != old_step):
            DataPoints.append([])
            old_step = step
            idx = 0
            if(max_steps != -1):
                if(step > max_steps):
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
            weights[set_idx].append([data[d][col + (i+1)*eigenvalues] for i in range(eigenvalues)])
            index[set_idx].append([col + (i+1)*eigenvalues for i in range(eigenvalues)])
            
        #if state_headers != []
        ## do some stuff

            DataPoints[step][idx].append(DataPoint([x], [eigenvalue_value[set_idx][eig_idx]], eig, 1.0, [colours_dict[eig_idx]],[state_headers[eig]]))
        idx += 1
    return DataPoints



def GetStateHeader(lineheader, eigenvalues, ignore_indices, offset):
    state_headers = []
    for i in range(0,eigenvalues):
        if i in ignore_indices:
            continue
        state_headers.append(lineheader[offset + ((i+1) * eigenvalues)])
    return state_headers

def assign_state_to_colour(data_points, state_headers):
    state_to_colour = {}
    all_states = False
    num_states_found = 0
    num_states = len(data_points[0])
    state_to_colour['mixed'] = (0.0,0.0,0.0,1.0)
    for DP in data_points:
        for dp in DP:
            if dp.eigenvalues == []:
                continue
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
            xs = [o for d in DataPoints[p] for i in d[s*eig:(s+1)*eig] for o in i.x]
            ys = [o for d in DataPoints[p] for i in d[s*eig:(s+1)*eig] for o in i.eigenvalues]
            cs = [o for d in DataPoints[p] for i in d[s*eig:(s+1)*eig] for o in i.colour]
            #state_dict = assign_state_to_colour(DataPoints, lineheader)
            ax.scatter(xs, ys, color=cs, s=20, edgecolors='none')

            #legend_elements = [
            #Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=10, label=legend_label) for legend_label, color in state_dict.items()
            #]
            #ax.legend(handles=legend_elements, title="Eigenstates", loc='upper right')

            ax.set_xlabel('Time (ns))')
            ax.set_ylabel('Eigenvalues (rad/ns)')
            plt.show()


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
    state_headers = ["GS.T0_U", "GS.T0_Z", "GS.T0_D", "GS.TP_U", "GS.TP_Z", "GS.TP_D", "GS.TD_U", "GS.TD_Z", "GS.TD_D", "ES.T0_U", "ES.T0_Z", "ES.T0_D", "ES.TP_U", "ES.TP_Z", "ES.TP_D", "ES.TD_U", "ES.TD_Z", "ES.TD_D"]
    Data = lines[1:len(lines)]
    DataPoints = CollectEigenstatesTR(Data, eigenvalues, 4, 1, sets=2, state_headers=state_headers)
    PlotTREigenStates(DataPoints,state_headers, sets=2,eig=eigenvalues)





#main()
PlotTimeResolvedEigenStates()


