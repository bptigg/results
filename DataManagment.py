import numpy as np
import mmap
from collections import defaultdict
import os
import time 
import threading
import paramiko
import io
import socket

class DataPointCollection:
    def __init__(self,filename,_internal_data=None,_internal_groups=None, _internal_indices=None):
        if _internal_data is not None:
            self.data = _internal_data
            self._group_map = _internal_groups
            self.derivative = True
            self.active_indices = _internal_indices
            self.cached_keys = []
        else:
            with open(filename, "rb") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    try:
                        os.madvise(mm, mmap.MADV_WILLNEED)
                        os.madvise(mm, mmap.MADV_SEQUENTIAL)
                    except (AttributeError, OSError):
                        pass
            self.data = np.load(filename, allow_pickle=True, mmap_mode='r')
            self._group_map = self.data['_group_map']
            self.derivative = False
            self.active_indices = np.arange(len(self._group_map))
        
            self.cached_keys = [k for k in self.data.files if k != '_group_map']
        
        self._index_map = defaultdict(list)
        for idx in self.active_indices:
            group_id = self._group_map[idx]
            self._index_map[group_id].append(idx)
        for group_id in self._index_map:
            self._index_map[group_id] = np.array(self._index_map[group_id])

        self.unique_groups = list(self._index_map.keys())
        self.num_groups = len(self.unique_groups)

        self.last_accessed_group = 0
        self.start_background_processing(lookahead=20)
        
    def __len__(self):
        return self.num_groups
    def __getitem__(self, key):
        if isinstance(key, slice):
            if not self.derivative and self.num_groups != 1:
                selected_groups = self.unique_groups[key]
                new_indices = np.concatenate([self._index_map[group_id] for group_id in selected_groups])
                #new_groups = self._group_map[mask]
                return DataPointCollection(None, _internal_data=self.data, _internal_groups=self._group_map, _internal_indices=new_indices)
            else:
                raise ValueError("Slicing is only supported for collections with a single group or when derivative is False.")
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Tuple key must have exactly two elements: (group_slice, data_slice).")
            group_idx, data_slice = key
            if isinstance(group_idx, int) and isinstance(data_slice, slice):
                if group_idx >= self.num_groups:
                    raise IndexError("Group index out of range.")
                self.last_accessed_group = group_idx
                group_id = self.unique_groups[group_idx]
                indices = self._index_map[group_id]
                sliced_indices = indices[data_slice]
                return DataPointCollection(None, _internal_data=self.data, _internal_groups=self._group_map, _internal_indices=sliced_indices)
            if isinstance(group_idx, slice) and isinstance(data_slice, slice):
                selected_groups = self.unique_groups[group_idx]
                new_indices = []
                for g_id in selected_groups:
                    new_indices.extend(self._index_map[g_id][data_slice])
                return DataPointCollection(None,_internal_data = self.data, _internal_groups = self._group_map, _internal_indices = np.array(new_indices))
        if key >= self.num_groups:
            raise IndexError("Group index out of range.")
        if isinstance(key, int):
            self.last_accessed_group = key
            group_id = self.unique_groups[key]
            indices = self._index_map[group_id]
            return [DataView(self.data, self.GetKeys(), i) for i in indices]
    def UniqueHeaders(self):
        #current_headers = self.data["Header"][np.isin(range(len(self.data["Header"])), self.active_indices)]
        current_headers = self.data["Header"][self.active_indices]
        num = len(np.unique(current_headers))
        return [num for _ in range(len(self))]
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    def GetKeys(self):
        return self.cached_keys
    
    def start_background_processing(self, lookahead=10, delay = 0.1, max_warm = 500):
        if self.derivative:
            return
        
        self.last_accessed_group = 0
        def streamer(max_warm=max_warm):
            warmed_up = set()
            while True:
                start = self.last_accessed_group
                end = min(start + lookahead, self.num_groups)
                for i in range(start,end):
                    if i not in warmed_up:
                        self.warm_group(i)
                        warmed_up.add(i)
                if len(warmed_up) > max_warm:
                    warmed_up = {idx for idx in warmed_up if idx >= start and idx < end}
                time.sleep(delay)
        t = threading.Thread(target=streamer, daemon=True)
        t.start()
        return t
    def warm_group(self, group_idx):
        group_id = self.unique_groups[group_idx]
        indices = self._index_map[group_id]
        for key in self.cached_keys:
            _ = self.data[key][indices]

        

class DataView:

    __slots__ = ['master', 'i', 'keys', '_cache']

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
             

def read_file(filename):
    if isinstance(filename,str):
        with open(filename, "rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                lines = []
                for line in iter(mm.readline, b""):
                    lines.append(line.decode('utf-8').split())
                return lines
    elif isinstance(filename,io.BytesIO):
        lines = []
        for line in filename:
            lines.append(line.decode('utf-8').split())
        return lines

        
def save_data_npz(header_names,lines, filename):
    #header_names = lines[0]
    data_rows = np.array(lines[1:])
    save_dict = {}
    save_dict['_group_map'] = data_rows[:,0]
    for i,name in enumerate(header_names):
        if i == 0: continue
        col_data = data_rows[:,i]
        try:
            save_dict[name] = col_data.astype(float)
        except ValueError:
            save_dict[name] = col_data
    np.savez(filename, **save_dict)
    print(f"Saved {filename} to .npz file")

def ProcessFile(filenames,new_filenames = [""]):
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(new_filenames, str):
        new_filenames = [new_filenames]
    for new_filenames in new_filenames:
        if new_filenames == "":
            new_filenames = [filenames[i] for i in range(len(filenames))]
    all_data_rows = []
    master_headers = None
    for i, fname in enumerate(filenames):
        f = fname + ".dat"
        lines = read_file(f)
        if not lines:
            print(f"Warning: File {f} is empty or could not be read.")
            continue
        header_names = lines[0]
        data_rows = np.array(lines[1:])
        data_rows[:,0] = np.char.add(data_rows[:,0].astype(str), f"_f{i}")
        if i == 0:
            master_headers = header_names
        all_data_rows.append(data_rows)
    if master_headers is None:
        print("Error: No valid files were processed.")
        return None
    if all_data_rows:
        combined_data = np.vstack(all_data_rows)
        save_data_npz(master_headers, combined_data, new_filenames[0])
        return new_filenames[0] + ".npz"
    else:
        print("Error: No valid data rows were found in the files.")
        return None


def ProcessFileSSH(filename, new_filename = "", dict_path = "",host = "", user = "", password = None, key_path = None):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname=host, username=user, password=password,key_filename=key_path,look_for_keys=True,timeout=20)
        print("connected")
        with ssh.open_sftp() as sftp:
            if(new_filename == ""):
                new_filename = filename
            remote_path = filename + ".dat"
            file_buffer = io.BytesIO()
            #print(sftp.listdir('.'))
            if(dict_path):
                sftp.chdir(dict_path)
            sftp.getfo(remotepath=remote_path,fl=file_buffer)
            file_buffer.seek(0)
            data = read_file(file_buffer)
            save_data_npz(data, new_filename)
            return new_filename + ".npz", True
    except paramiko.AuthenticationException:
        print("Error: Authentication failed. Please check your username/password.")
        #return None
    except paramiko.SSHException as ssh_err:
        print(f"Error: SSH connection failed: {ssh_err}")
        #return None
    except socket.error as sock_err:
        print(f"Error: Network unreachable or timeout: {sock_err}")
        #return None
    except(paramiko.SSHException, socket.error) as e:
        print(f"Connection failed {e}")
    finally:
        ssh.close()

    print("Script will open npz file of same name, if this is not the first time it has been run")
    return False




class ExponentiallyWeightedMovingAverage:
    def __init__(self, tau, mean = None):
        self.tau = tau
        self.mean = mean
        self.var = 0.0

        self.tau2 = tau * 100
        self.mean_std = None
        self.var_std = 0.0
    def update(self,current_val,dt):
        alpha = 1.0 - np.exp(-dt / self.tau)
        alpha2 = 1.0 - np.exp(-dt / self.tau2)
        if(self.mean is None):
            self.mean = current_val
            self.var = 0.0
            return 0.0, self.mean, 0.0, self.mean_std
        
        diff = current_val - self.mean
        self.mean += alpha * diff
        self.var = (1.0 - alpha) * (self.var + alpha * diff**2)
        current_std = np.sqrt(max(0.0,self.var))

        if(self.mean_std is None):
            self.mean_std = current_std
        else:
            diff_std = current_std - self.mean_std
            self.mean_std += alpha * diff_std
            self.var_std = (1.0 - alpha2) * (self.var_std + alpha2 * diff**2)
        std_dev_of_std = np.sqrt(max(0.0,self.var_std))

        return current_std, self.mean ,std_dev_of_std, self.mean_std
