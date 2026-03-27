#produce MARY curve from NV-TimeEvoData

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import mmap

class DataPoint:
    def __init__ (self, B, population, Header, color, secondary_coord=0.0):
        self.B = B
        self.population = population
        self.Header = Header
        self.color = color
        self.secondary_coord = secondary_coord

class DataPointCollection:
    def __init__(self,filename):
        self.data = np.load(filename, allow_pickle=True, mmap_mode='r')
        self._group_map = self.data['_group_map']
        self.unique_groups = np.unique(self._group_map)
        self.cached_keys = [k for k in self.data.files if k != '_group_map']

    def __len__(self):
        return len(self.unique_groups)
    def __getitem__(self, key):
        indices = np.where(self._group_map == key)[0]
        return [DataView(self.data, self.GetKeys(), i) for i in indices]
    def UniqueHeaders(self):
        num = len(np.unique(self.data["Header"]))
        return [num for _ in range(len(self))]
    def GetKeys(self):
        return self.cached_keys

class DataView:
    def __init__(self, master_collection,keys, index):
        self.master = master_collection
        self.i = index
        self.keys = keys
        self._cache = {}
    
    ##@property
    ##def B(self):
    ##    return self.master[self.keys[0]][self.i]
    ##@property
    ##def population(self):
    ##    return self.master[self.keys[1]][self.i]
    ##@property
    ##def Header(self):
    ##    return self.master[self.keys[2]][self.i]
    ##@property
    ##def color(self):
    ##    return self.master[self.keys[3]][self.i]
    ##@property
    ##def secondary_coord(self):
    ##    return self.master[self.keys[4]][self.i]

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
             

def read_file(filename):
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            lines = []
            for line in iter(mm.readline, b""):
                lines.append(line.decode('utf-8').split())
            return lines

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
        for i in range(offset,len(data[d])):
            if(selected_data == []):
                collected_data.append(DataPoint(float(B), float(data[d][i]), headers[i], colours_dict[i-offset], float(sc)))
            elif headers[i] in selected_data:
                collected_data.append(DataPoint(float(B), float(data[d][i]), headers[i], colours_dict[i-offset], float(sc)))

    points = len(selected_data)
    d = len(data) - 1
    B = data[d][x_coord]
    for i in range(offset,len(data[d])):
        if(selected_data == []):
            collected_data.append(DataPoint(float(B), float(data[d][i]), headers[i], colours_dict[i-offset],float(sc)))
        elif headers[i] in selected_data:
            collected_data.append(DataPoint(float(B), float(data[d][i]), headers[i], colours_dict[i-offset],float(sc)))

    return collected_data, points

def assign_state_to_color(data_points):
    state_to_colour = {}
    for DP in data_points:
        state = DP.Header
        color = DP.color
        state_to_colour[state] = color
    return state_to_colour

def GetDimension(data):
    if type(data) == DataPoint or type(data) == DataView:
        return []
    return [len(data)] + GetDimension(data[0])

def load_block(collection : DataPointCollection, idx):
    mask = collection._group_map ==collection.unique_groups[idx]
    dat_x = collection.data['B'][mask]
    dat_z = collection.data['population'][mask]
    dat_y = collection.data['secondary_coord'][mask]
    dat_c = collection.data['color'][mask]

    return dat_x,dat_y,dat_z,dat_c
 
def PlotStabalizedPopulation3D(stab_pop, num, dim):
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')

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
        full_x = np.concatenate((full_x,x))
        full_y = np.concatenate((full_y,y))
        full_z = np.concatenate((full_z,z))
        if type(full_c) == list:
            full_c = c
        else:
            full_c = np.concatenate((full_c,c))


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
        


def PlotStabalizedPopulation(stab_pop, num):
    dim = GetDimension(stab_pop)
    if(len(dim) > 1):
        PlotStabalizedPopulation3D(stab_pop,num, dim)
        return
    fig, ax = plt.subplots()
    xs = [d.B for d in stab_pop]
    ys = [d.population for d in stab_pop]
    cs = [d.color for d in stab_pop]

    state_dict = assign_state_to_color(stab_pop[0:num])
    ax.scatter(xs,ys,color=cs, s=20, edgecolors='none')
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=10, label=legend_label) for legend_label, color in state_dict.items()
    ]
    ax.legend(handles=legend_elements, title="states", loc='upper right')

    plt.show()

def PlotPhotoLuminesance(stab_pop, num):
    fig, ax = plt.subplots()
    xs = [stab_pop[d].B for d in range(0,len(stab_pop),num)]
    ys = [d.population for d in stab_pop]
    #cs = [d.color for d in stab_pop]

    pl = []
    for i in range(0,len(ys), num):
        sum = 0
        for e in range(0,num):
            sum += ys[i+e]
        pl.append(sum)
    
    ax.scatter(xs,pl, s=20, edgecolors='none')
    plt.show()

def save_stabalised_population(filename, stab_pop):
    flat_list = []
    group_ids = []
    for group_idx, sub_list in enumerate(stab_pop):
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
    #data = read_file("NV_centre_N14_GSLAC-SS-3.dat")
    #headers = data[0]
    #stab_pop, num = CollectStabalizedPopulations(data,headers,[2,3],4,['es.t0','es.t0_u', 'es.t0_z', 'es.t0_d', 'es.tp_u', 'es.tp_z', 'es.tp_d', 'es.td_u', 'es.td_z', 'es.td_d'])
    #save_stabalised_population("NV_N14_B(0.5-1.5)_sweep", stab_pop)

    stab_pop,num = load_stabalised_population("NV_N14_B(0.5-1.5)_sweep")
    #num =[9]
    
    #stab_pop, num = CollectStabalizedPopulations(data,headers,3,4,['gs.t0','gs.t0_u', 'gs.t0_z', 'gs.t0_d', 'gs.tp_u', 'gs.tp_z', 'gs.tp_d', 'gs.td_u', 'gs.td_z', 'gs.td_d'])
    PlotStabalizedPopulation(stab_pop, num)
    #PlotPhotoLuminesance(stab_pop, num)

if __name__ == "__main__":
    main()