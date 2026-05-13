import numpy as np
import mmap
from collections import defaultdict
import os
import time 
import threading
import paramiko
import io
import socket
import sys
import logging


class DataPointCollection:
    #def __init__(self,filename,_internal_data=None,_internal_groups=None, _internal_indices=None):
    def __init__(self, filenames=None, _internal_state=None):
        if _internal_state is not None:
            self.files = _internal_state['files']
            self._group_map = _internal_state['group_map']
            self.active_indices = _internal_state['indices']
            self.file_row_offsets = _internal_state['row_offsets']
            self.cached_keys = _internal_state['keys']
            self.derivative = True
        else:
            self.files = []
            self.file_row_offsets = [0]
            self.derivative = False
            global_group_map = []
            current_group_offset= 0
            if(isinstance(filenames, str)):
                filenames = [filenames]
            for filename in filenames:
                self._mmap_warm(filename)
                data = np.load(filename, allow_pickle=True, mmap_mode='r')
                self.files.append(data)
                local_groups_raw = data['_group_map']
                try:
                    local_groups = local_groups_raw.astype('int64')
                except ValueError:
                    _, local_groups = np.unique(local_groups_raw, return_inverse=True)
                offset_groups = local_groups + current_group_offset
                global_group_map.append(offset_groups)
                current_group_offset = offset_groups.max() + 1
                self.file_row_offsets.append(self.file_row_offsets[-1] + len(local_groups))
            self._group_map = np.concatenate(global_group_map)
            self.derivative = False
            self.active_indices = np.arange(len(self._group_map))
        
            self.cached_keys = [k for k in self.files[0].files if k != '_group_map']
        
        self._index_map = defaultdict(list)
        for idx in self.active_indices:
            self._index_map[self._group_map[idx]].append(idx)
        for group_id in self._index_map:
            self._index_map[group_id] = np.array(self._index_map[group_id])

        self.unique_groups = sorted(list(self._index_map.keys()))
        self.num_groups = len(self.unique_groups)

        self.last_accessed_group = 0
        self.start_background_processing(lookahead=20)
        self.force_groups = False

    def _mmap_warm(self, filename):
        try:
            with open(filename, "rb") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    os.madvise(mm, mmap.MADV_WILLNEED)
                    os.madvise(mm, mmap.MADV_SEQUENTIAL)
        except(AttributeError,OSError,ValueError):
            pass
    def _get_data_from_global_idx(self, key, global_idx):
        if len(self.files) == 1:
            return self.files[0][key][global_idx]
        if isinstance(global_idx, int):
            global_idx = int(global_idx)
            file_idx = np.searchsorted(self.file_row_offsets, global_idx, side='right') -1
            local_idx = global_idx - self.file_row_offsets[file_idx]
            return self.files[file_idx][key][local_idx]
        else:
            idx_array = np.asanyarray(global_idx)
            f_idx = np.searchsorted(self.file_row_offsets, global_idx, side='right') - 1
            first_file_dtype = self.files[0][key]
            result = np.empty(len(idx_array), dtype=first_file_dtype.dtype)
            unique_f_ids = np.unique(f_idx)
            for f_id in unique_f_ids:
                mask = (f_idx == f_id)
                local = idx_array[mask] - self.file_row_offsets[f_id]
                result[mask] = self.files[f_id][key][local]
            return result
    
    @property
    def data(self):
        return MultiFileProxy(self)
    
    def __len__(self):
        if self.num_groups == 1 and not self.force_groups:
            return len(self.active_indices)
        self.force_groups = False
        return self.num_groups
    def __getitem__(self, key):
        if isinstance(key, slice):
            if not self.derivative and self.num_groups != 1:
                selected_groups = self.unique_groups[key]
                new_indices = np.concatenate([self._index_map[group_id] for group_id in selected_groups])
            elif self.num_groups == 1:
                selected_groups = self.unique_groups[0]
                new_indices = self._index_map[selected_groups][key]
            else:
                raise ValueError("Slicing is only supported for collections with a single group or when derivative is False.")
            state = {
                'files' : self.files,
                'group_map' : self._group_map,
                'indices' : np.array(new_indices),
                'row_offsets' : self.file_row_offsets,
                'keys' : self.cached_keys
            }
            return DataPointCollection(_internal_state = state)
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Tuple key must have exactly two elements: (group_slice, data_slice).")
            group_idx, data_slice = key
            if isinstance(group_idx, int):
                if group_idx >= self.num_groups:
                    raise IndexError("Group index out of range.")
                selected_groups = [self.unique_groups[group_idx]]
                self.last_accessed_group = group_idx
            elif isinstance(group_idx, slice):
                g_start, g_stop, g_step = group_idx.indices(self.num_groups)
                if g_stop >= self.num_groups+1:
                    raise IndexError("Group index out of range.")
                selected_groups = self.unique_groups[group_idx]
            else:
                raise TypeError("Group index must be int or slice")
            new_indices = []
            for g_id in selected_groups:
                group_indices = self._index_map[g_id]
                new_indices.extend(group_indices[data_slice])
            
            state = {
                'files' : self.files,
                'group_map' : self._group_map,
                'indices' : np.array(new_indices),
                'row_offsets' : self.file_row_offsets,
                'keys' : self.cached_keys
            }

            return DataPointCollection(_internal_state = state)
        
        if isinstance(key, int):
            if key >= self.num_groups:
                raise IndexError("Group index out of range.")
            self.last_accessed_group = key
            group_id = self.unique_groups[key]
            indices = self._index_map[group_id]
            return [DataView(self.data, self.GetKeys(), i) for i in indices]
    def UniqueHeaders(self):
        #current_headers = self.data["Header"][np.isin(range(len(self.data["Header"])), self.active_indices)]
        current_headers = [self._get_data_from_global_idx("Header",idx) for idx in self.active_indices]
        num = len(np.unique(current_headers))
        return [num for _ in range(len(self))]
    def __iter__(self):
        self.force_groups = True
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
        for idx in indices:
            f_idx = np.searchsorted(self.file_row_offsets, idx, side='right') - 1
            local_idx = idx - self.file_row_offsets[f_idx]
            for key in self.cached_keys:
                _ = self.files[f_idx][key][local_idx]

class DataView:

    __slots__ = ['master', 'i', 'keys', '_cache']

    def __init__(self, master_collection : DataPointCollection,keys, index):
        self.master = master_collection
        self.i = index
        self.keys = keys
        self._cache = {}

    def __getitem__(self, key):
        return self.master._get_data_from_global_idx(key, self.i)

    def __getattr__(self, name):
        if name in self._cache:
            return self._cache[name]
        
        if name in self.keys:
            val = self.master._get_data_from_global_idx(name, self.i)
            if hasattr(val,'item') and not isinstance(val, np.ndarray):
                val = val.item()
            self._cache[name] = val
            return val
        raise AttributeError(f"'DataView' object has no attribute '{name}'")
    
    def __repr__(self):
        try:
            header_val = self.Header
        except AttributeError:
            header_val = "Unkown"
        return f"<DataView index={self.i} header={header_val}>"
             

def read_file(filename):
    stats = {'last_bytes': 0, 'last_time': time.time(), 'current_speed': 0.0}
    
    def update_ui(current_bytes, total_bytes):
        current_time = time.time()
        time_diff = current_time - stats['last_time']
        if time_diff >= 0.5:
            bytes_diff = current_bytes - stats['last_bytes']
            stats['current_speed'] = (bytes_diff / (1024**2)) / time_diff
            stats['last_bytes'] = current_bytes
            stats['last_time'] = current_time
            
            percent = (current_bytes / total_bytes) * 100
            msg = f"\rParsing Data: {percent:5.1f}% | Speed: {stats['current_speed']:6.2f} MB/s"
            sys.stdout.write(msg)
            sys.stdout.flush()
    if isinstance(filename,str):
        filesize = os.path.getsize(filename)
        with open(filename, "rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                lines = []
                for line in iter(mm.readline, b""):
                    lines.append(line.decode('utf-8').split())
                    if len(lines) % 1000 == 0:
                        update_ui(mm.tell(), filesize)
                print()
                return lines
    elif isinstance(filename,io.BytesIO):
        lines = []
        for line in filename:
            lines.append(line.decode('utf-8').split())
        return lines

def peak_to_end_of_file(filename):
    with open(filename, 'r+b') as f:
        try:
            f.seek(0, os.SEEK_END)
            pointer = f.tell()
            buffer = b""
            while pointer > 0:
                pointer -= 1
                f.seek(pointer)
                char = f.read(1)
                if char == b'\n' and buffer:
                    break
                buffer = char + buffer
            return buffer.decode('utf-8').strip()
        except OSError:
            return None

class MultiFileProxy:
    def __init__(self, master):
        self.master = master
    def __getitem__(self, key):
        return KeyProxy(self.master, key)
    def _get_data_from_global_idx(self, name, idx):
        return self.master._get_data_from_global_idx(name, idx)


class KeyProxy:
    def __init__(self,master : DataPointCollection, key):
        self.master =  master
        self.key = key
    def __getitem__(self, indices):
        if isinstance(indices, int):
            return self.master._get_data_from_global_idx(self.key, indices)
        if(len(self.master.files)) == 1:
            return self.master.files[0][self.key][indices]
        return np.array(self.master._get_data_from_global_idx(self.key, indices))

#def concatinate_multiple_npz_files(base_file, start_idx, end_idx):
    


        
def save_data_npz(header_names,lines, filename):
    #header_names = lines[0]
    data_rows = np.array(lines[1:])
    save_dict = {}
    save_dict['_group_map'] = data_rows[:,0]
    for i,name in enumerate(header_names):
        percent = int(((i+1) / len(header_names)) * 100)
        bar_length = 20
        filled_length = int(bar_length * (i+1) // len(header_names))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f"\rSaving Data: |{bar}| {percent}% ({name[:15]})")
        sys.stdout.flush()

        if i == 0: continue
        col_data = data_rows[:,i]
        try:
            save_dict[name] = col_data.astype(float)
        except ValueError:
            save_dict[name] = col_data
    np.savez(filename, **save_dict)
    print(f"Saved {filename} to .npz file")

def BatchProcess(base_file, idx_start, idx_end, new_base_filename = "", base_path = ""):
    filenames = []
    new_filenames = []
    for i in range(idx_start, idx_end+1):
        base_filename = ""
        if base_path:
            base_filename = os.path.join(base_path, base_file)
        else:
            base_filename = base_file
        filenames.append(f"{base_filename}-{i}")
        if new_base_filename:
            new_filenames.append(f"{new_base_filename}-{i}")
        else:
            new_filenames.append(base_file + f"-{i}")
    BatchProcessFiles(filenames, new_filenames)

def BatchProcessFiles(filenames, new_filenames = [""]):
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(new_filenames, str):
        new_filenames = [new_filenames]
    if len(new_filenames) == 1 and new_filenames[0] == "":
        new_filenames = [filenames[i] for i in range(len(filenames))]
    for i, fname in enumerate(filenames):
        f = fname + ".dat"
        lines = read_file(f)
        if not lines:
            print(f"Warning: File {f} is empty or could not be read.")
            continue
        header_names = lines[0]
        save_data_npz(header_names, lines[1:], new_filenames[i])

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

def enroll_ssh_key(ssh_client, key_path):
    if not os.path.exists(key_path):
        print(f"Generating new RSA key at {key_path}...")
        key = paramiko.RSAKey.generate(2048)
        key.write_private_key_file(key_path)
    else:
        key = paramiko.RSAKey.from_private_key_file(key_path)

    public_key_str = f"{key.get_name()} {key.get_base64()}"
    
    # Ensure remote .ssh directory exists and append the public key
    setup_cmd = f'mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "{public_key_str}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
    stdin, stdout, stderr = ssh_client.exec_command(setup_cmd)
    
    if stdout.channel.recv_exit_status() == 0:
        print("Successfully enrolled local key on the remote server.")
        return True
    return False


def ProcessFileSSH(filename, new_filename = "", dict_path = "",host = "", user = "", password = None, key_path = "~/.ssh/id_rsa_generated"):
    ssh = paramiko.SSHClient()
    known_hosts_path = os.path.expanduser("~/.ssh/known_hosts")
    os.makedirs(os.path.dirname(known_hosts_path), exist_ok=True)
    if(os.path.exists(known_hosts_path)):
        ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    full_key_path = os.path.expanduser(key_path)
    logging.basicConfig(level=logging.WARNING)
    try:
        try:
            my_key = paramiko.RSAKey.from_private_key_file(full_key_path)
            ssh.connect(hostname=host, username=user, pkey=my_key, look_for_keys=False, allow_agent=False, timeout=20)
            #ssh.connect(hostname=host, username=user, key_filename=key_path,look_for_keys=False,allow_agent=False,timeout=20)
            print(f"Connected to {host} via SSH Key.")
        except (paramiko.AuthenticationException, paramiko.SSHException, FileNotFoundError):
            if password:
                print("Key auth failed. Attempting password login and key enrollment...")
                ssh.connect(hostname=host, username=user, password=password, timeout=15)
                
                ssh.save_host_keys(known_hosts_path)
                enroll_ssh_key(ssh, full_key_path)
            else:
                print("No valid key found and no password provided.")
                return False
        with ssh.open_sftp() as sftp:
            if(new_filename == ""):
                new_filename = filename
            remote_path = filename + ".dat"
            file_buffer = io.BytesIO()
            print(sftp.listdir('.'))
            if(dict_path):
                sftp.chdir(dict_path)

            start = time.time()
            stats = {
                'last bytes' : 0,
                'last time' : start,
                'current speed' : 0.0
            }

            def ProgressStats(transferred, total):
                currenttime = time.time()
                time_diff = currenttime - stats['last time']
                if time_diff >= 1.0:
                    bytes_diff = transferred - stats['last bytes']
                    stats['current speed'] = (bytes_diff / (1024**2)) / time_diff
                    stats['last bytes'] = transferred
                    stats['last time'] = currenttime
                percent = (transferred / total) * 100
                done_mb = transferred / (1024**2)
                total_mb = total / (1024**2)
                msg = f"\rProgress: {percent:5.1f}% | Speed: {stats['current speed']:6.2f} MB/s | {done_mb:7.1f}/{total_mb:.1f} MB"
                sys.stdout.write(msg)
                sys.stdout.flush()

            sftp.getfo(remotepath=remote_path,fl=file_buffer,callback=ProgressStats)
            print()
            time_diff = time.time() - start
            mins,secs = divmod(int(time_diff),60)
            msg = f"\rTotal time: {mins}m:{secs}s"
            sys.stdout.write(msg)
            sys.stdout.flush()
            file_buffer.seek(0)
            data = read_file(file_buffer)
            headers = data[0]
            save_data_npz(headers, np.array(data[1:]), new_filename)
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
