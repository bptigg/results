#produce MARY curve from NV-TimeEvoData

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import mmap
import DataManagment as dm
from DataManagment import DataPointCollection, DataView

class DataPoint:
    def __init__ (self, B, population, Header, color, secondary_coord=0.0):
        self.B = B
        self.population = population
        self.Header = Header
        self.color = color
        self.secondary_coord = secondary_coord

cmp_init = False
colours_dict ={}
def StartColourMap():
    global cmp_init
    if cmp_init:
        return
    
    cmap = mpl.colormaps['tab20']
    cmapb = mpl.colormaps['tab20b']
    cmapc = mpl.colormaps['tab20c']
    colors = cmap.colors
    colors2 = cmapb.colors
    colors3 = cmapc.colors
    cmp_init = True
    for i, colour in enumerate(colors):
        colours_dict[i] = (colour[0], colour[1], colour[2], 1.0)
    for i, colour in enumerate(colors2):
        colours_dict[i+20] = (colour[0], colour[1], colour[2], 1.0)
    for i, colour in enumerate(colors3):
        colours_dict[i+40] = (colour[0], colour[1], colour[2], 1.0)

def Internal_CollectStabalizedPopulationsMultipleAxis(data, headers, axis, offset, selected_data):
    #split the data into groupings of x_coord
    old_step = data[1][axis[0]]
    new_data = []
    sorted_data = []
    points = []
    for d in range(1,len(data)):
        step = data[d][axis[0]]
        if(step == old_step):
            new_data.append(data[d])
        else:
            cd, p = CollectStabalizedPopulations(new_data,headers,axis[1],offset,selected_data, old_step)
            old_step = step
            sorted_data.append(cd)
            points.append(p)
            new_data = []
    
    if(new_data != []):
        cd, p = CollectStabalizedPopulations(new_data,headers,axis[1],offset,selected_data, old_step)
        sorted_data.append(cd)
        points.append(p)
        new_data = []
    return sorted_data, points
            

def CollectStabalizedPopulations(data, headers, x_coord, offset, selected_data = [], sc = 0.0):
    old_step = data[1][0]
    collected_data = []
    StartColourMap()
    points = 0
    if(isinstance(x_coord,list)):
        return Internal_CollectStabalizedPopulationsMultipleAxis(data,headers,x_coord,offset,selected_data)
    for d in range(1,len(data)):
        step = data[d][0]
        if(step == old_step):
            continue
        old_step = step
        d = d - 1
        B = float(data[d][x_coord])
        collected_data.append([])
        in_head = []
        for i in range(offset,len(data[d])):
            if(selected_data == []):
                collected_data[int(step)-2].append(DataPoint(float(B), float(data[d][i]), headers[i], colours_dict[i-offset], float(sc)))
            elif headers[i] in selected_data:
                collected_data[int(step)-2].append(DataPoint(float(B), float(data[d][i]), headers[i], colours_dict[i-offset], float(sc)))
                in_head.append(headers[i])
        print(in_head)

    points = len(selected_data)
    d = len(data) - 1
    B = data[d][x_coord]
    collected_data.append([])
    step = data[d][0]
    for i in range(offset,len(data[d])):
        if(selected_data == []):
            collected_data[int(step)-1].append(DataPoint(float(B), float(data[d][i]), headers[i], colours_dict[i-offset],float(sc)))
        elif headers[i] in selected_data:
            collected_data[int(step)-1].append(DataPoint(float(B), float(data[d][i]), headers[i], colours_dict[i-offset],float(sc)))

    return collected_data, points

def assign_state_to_color(data_points):
    state_to_colour = {}
    for d in data_points:
        for DP in d:
            state = DP.Header
            color = DP.color
            state_to_colour[state] = color
    return state_to_colour

def GetDimension(data):
    if type(data) == DataPoint or type(data) == DataView:
        return []
    return [len(data)] + GetDimension(data[0])

def load_block(collection : DataPointCollection, idx):
    collection.last_accessed_group = idx
    group_id = collection.unique_groups[idx]
    indicies = collection._index_map[group_id]
    dat_x = collection.data['B'][indicies]
    dat_z = collection.data['population'][indicies]
    dat_y = collection.data['secondary_coord'][indicies]
    dat_c = collection.data['color'][indicies]

    return dat_x,dat_y,dat_z,dat_c
 
def PlotStabalizedPopulation3D(stab_pop, num, dim, slice = [-1,-1]):
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    if(slice == [-1,-1]):
        slice = [0,num[0]]

    state_dict = assign_state_to_color(stab_pop[0][0:num[0]])
    #legend_elements = [
    #    Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=10, label=legend_label) for legend_label, color in state_dict.items()
    #]
    #ax.legend(handles=legend_elements, title="states", loc='upper right')

    full_x = []
    full_y = []
    full_z = []    
    full_c = []
    for i in range(0,dim[0]):
        x,y,z,c = load_block(stab_pop,i)
        full_x = np.concatenate((full_x,x[slice[0]:slice[1]]))
        full_y = np.concatenate((full_y,y[slice[0]:slice[1]]))
        full_z = np.concatenate((full_z,z[slice[0]:slice[1]]))
        if type(full_c) == list:
            full_c = c[slice[0]:slice[1]]
        else:
            full_c = np.concatenate((full_c,c[slice[0]:slice[1]]))


    for i in range(0,num[0]):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        legend_elements = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=10, label=legend_label) for legend_label, color in state_dict.items()
        ]
        ax.legend(handles=legend_elements, title="states", loc='upper right')

        #xs = [d.B for sub in stab_pop for d in sub[i::num[0]]]
        #zs = [d.population for sub in stab_pop for d in sub[i::num[0]]]
        #ys = [float(d.secondary_coord) for sub in stab_pop for d in sub[i::num[0]]]
        #cs = [d.color for sub in stab_pop for d in sub[i::num[0]]]

        xs = [d for d in full_x[i::num[0]]]
        ys = [d for d in full_y[i::num[0]]]
        zs = [d for d in full_z[i::num[0]]]


        ax.scatter(xs, ys, zs, c=zs, cmap = "magma", s= 50, marker='o')
        ax.set_xlim(min(xs),max(xs))
        ax.set_ylim(min(ys),max(ys))
        ax.set_zlim(min(zs),max(zs))
        #ax.set_zlim(0.034,0.036)
        ax.view_init(-90,0,0)
        #print("state = {}".format(state_dict[i]))

        plt.show()


    
        #ax.scatter(xs, ys, zs, c=cs, s= 20, marker='o',edgecolors ='none')
        #ax.scatter(xs,zs,color=cs, s=20, edgecolors='none')
        #print(ys[0])

    #plt.show()
        


def PlotStabalizedPopulation(stab_pop, num, x_label = r"B$_{||}$ (T)", y_label = "state population", slice = [-1,-1]):
    dim = GetDimension(stab_pop)
    if(len(dim) > 2):
        PlotStabalizedPopulation3D(stab_pop,num, dim, slice=slice)
        return
    if(slice == [-1,-1]):
        slice = [0,num[0]]
    fig, ax = plt.subplots()
    xs = []
    ys = []
    cs = []
    for i in range(0,len(stab_pop)):
        x,y,z,c = load_block(stab_pop,i)
        xs = np.concatenate((xs,x[slice[0]:slice[1]]))
        ys = np.concatenate((ys,z[slice[0]:slice[1]]))
        if type(cs) == list:
            cs = c[slice[0]:slice[1]]
        else:
            cs = np.concatenate((cs,c[slice[0]:slice[1]]))
    #xs = [d.B for d in stab_pop]
    # = [d.population for d in stab_pop]
    #cs = [d.color for d in stab_pop]

    state_dict = assign_state_to_color(stab_pop[0, slice[0]:slice[1]])
    ax.scatter(xs,ys,color=cs, s=20, edgecolors='none')
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=10, label=legend_label) for legend_label, color in state_dict.items()
    ]
    ax.legend(handles=legend_elements, title="states", loc='upper right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.draw()

def PlotPhotoLuminesance(stab_pop, num, x_label = r"B$_{||}$ (T)", y_label = "normalized photoluminescence", slice = [0,3], k_decay = 1.0/12.0, k_isc = [], k_isc_index = []):
    fig, ax = plt.subplots()
    xs = []
    ys = []
    cs = []
    frac = []
    rate = 0
    for i in range(0,k_isc_index[len(k_isc_index)-1]-1):
        if i == k_isc_index[rate]-1:
            rate += 1
        frac.append(k_decay/(k_decay+k_isc[rate]))
    for i in range(0,len(stab_pop)):
        x,y,z,c = load_block(stab_pop,i)
        z = z * frac
        xs = np.concatenate((xs,[x[0]]))
        ys = np.concatenate((ys,z[slice[0]:slice[1]]))
        if type(cs) == list:
            cs = c
        else:
            cs = np.concatenate((cs,c))

    pl = []
    diff = slice[1] - slice[0]
    for i in range(0,len(ys), diff):
        sum = 0
        for e in range(0,diff):
            sum += ys[i+e]
        pl.append(sum)

    #rpl = np.array(pl)/max(pl)
    #normalized photoluminescence
    min_pl = min(pl)
    max_pl = max(pl)
    rpl = [(p - min_pl)/(max_pl - min_pl) for p in pl]
    
    ax.scatter(xs,rpl, s=20, edgecolors='none')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.draw()

def save_stabalised_population(filename, stab_pop):
    flat_list = []
    group_ids = []
    for group_idx, sub_list in enumerate(stab_pop):
        if(type(sub_list) != list): 
            sub_list = [sub_list]
        for p in sub_list:
            group_ids.append(group_idx)
            flat_list.append(p)

    sample_vars = flat_list[0].__dict__
    save_dict = {'_group_map': group_ids}

    for var_name in sample_vars:
        data = [getattr(p,var_name) for p in flat_list]
        save_dict[var_name] = np.array(data, dtype=object)

    np.savez_compressed(filename, **save_dict)
    print(f"Saved variables: {list(sample_vars)}")



def load_stabalised_population(filename):
    filename = filename + ".npz"
    dat = DataPointCollection(filename)
    return dat,dat.UniqueHeaders()

def main():
    data = dm.read_file("NV_centre_N14_GSLAC-PL-5.dat")
    headers = data[0]
    #stab_pop, num = CollectStabalizedPopulations(data,headers,[2,3],4,['es.t0_u', 'es.t0_z', 'es.t0_d', 'es.tp_u', 'es.tp_z', 'es.tp_d', 'es.td_u', 'es.td_z', 'es.td_d'])
    stab_pop, num = CollectStabalizedPopulations(data,headers,2,3,['es.t0_u', 'es.t0_z', 'es.t0_d', 'es.tp_u', 'es.tp_z', 'es.tp_d', 'es.td_u', 'es.td_z', 'es.td_d'])#, 'ms.i'])
    save_stabalised_population("NV_N14_GSLAC-PL-5", stab_pop)

    stab_pop,num = load_stabalised_population("NV_N14_GSLAC-PL-5")
    #num =[9]
    
    #stab_pop, num = CollectStabalizedPopulations(data,headers,3,4,['gs.t0','gs.t0_u', 'gs.t0_z', 'gs.t0_d', 'gs.tp_u', 'gs.tp_z', 'gs.tp_d', 'gs.td_u', 'gs.td_z', 'gs.td_d'])
    PlotStabalizedPopulation(stab_pop, num, slice = [1,num[0]])
    stab_pop2 = stab_pop[:, 0:num[0]-1]
    PlotPhotoLuminesance(stab_pop2, num,slice = [1,num[0]-1], k_decay = 1.0/13.0, k_isc = [0.001,0.083], k_isc_index = [4,num[0]])
    #plt.show()

    #stab_pop,num = load_stabalised_population("NV_N14_GSLAC-PL-4")
    #PlotStabalizedPopulation(stab_pop, num, slice = [1,num[0]])
    #stab_pop2 = stab_pop[:, 0:num[0]-1]
    #PlotPhotoLuminesance(stab_pop2, num,slice = [1,num[0]-1], k_decay = 1.0/13.0, k_isc = [0.001,0.083], k_isc_index = [4,num[0]])
    plt.show()

if __name__ == "__main__":
    main()