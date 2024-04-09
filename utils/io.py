import re 
import h5py
import numpy as np

# kaldi io operations are taken from 
# https://github.com/KarelVesely84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py#L487

# Define all 'kaldi_io' exceptions,
class UnsupportedDataType(Exception): pass
class UnknownVectorHeader(Exception): pass
class UnknownMatrixHeader(Exception): pass
class BadSampleSize(Exception): pass
class BadInputFormat(Exception): pass
class SubprocessFailed(Exception): pass

def _read_compressed_mat(fd, format):
    """ Read a compressed matrix,
        see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
        methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
    """
    assert(format == 'CM '), format # The formats CM2, CM3 are not supported...

    # Format of header 'struct',
    global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
    per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

    # Read global header,
    globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

    # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
    #                                                 {                     cols                     }{         size                 }
    col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
    col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers], dtype=np.float32)
    data = np.reshape(np.frombuffer(fd.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major,

    mat = np.zeros((cols,rows), dtype='float32')
    p0 = col_headers[:, 0].reshape(-1, 1)
    p25 = col_headers[:, 1].reshape(-1, 1)
    p75 = col_headers[:, 2].reshape(-1, 1)
    p100 = col_headers[:, 3].reshape(-1, 1)
    mask_0_64 = (data <= 64)
    mask_193_255 = (data > 192)
    mask_65_192 = (~(mask_0_64 | mask_193_255))

    mat += (p0    + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
    mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
    mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

    return mat.T # transpose! col-major -> row-major,

def read_key(fd):
    """ [key] = read_key(fd)
     Read the utterance-key from the opened ark/stream descriptor 'fd'.
    """
    if type(fd.mode) is str:
        assert('b' in fd.mode), "Error: 'fd' was opened in text mode (in python3 use sys.stdin.buffer)"

    key = ''
    while 1:
        char = fd.read(1).decode("latin1")
        if char == '' : break
        if char == ' ' : break
        key += char
    key = key.strip()
    if key == '': return None # end of file,
    assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
    return key

def _read_mat_binary(fd):
    # Data type
    header = fd.read(3).decode()
    # 'CM', 'CM2', 'CM3' are possible values,
    if header.startswith('CM'): return _read_compressed_mat(fd, header)
    elif header == 'FM ': sample_size = 4 # floats
    elif header == 'DM ': sample_size = 8 # doubles
    else: raise UnknownMatrixHeader("The header contained '%s'" % header)
    assert(sample_size > 0), sample_size
    # Dimensions
    s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
    # Read whole matrix
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4 : vec = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8 : vec = np.frombuffer(buf, dtype='float64')
    else : raise BadSampleSize
    mat = np.reshape(vec,(rows,cols))
    return mat

def _read_mat_ascii(fd):
    rows = []
    while 1:
        line = fd.readline().decode()
        if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
        if len(line.strip()) == 0 : continue # skip empty line
        arr = line.strip().split()
        if arr[-1] != ']':
            rows.append(np.array(arr,dtype='float32')) # not last line
        else:
            rows.append(np.array(arr[:-1],dtype='float32')) # last line
            mat = np.vstack(rows)
            return mat
        
def read_mat(fd):
    """ [mat] = read_mat(file_or_fd)
     Reads single kaldi matrix, supports ascii and binary.
     file_or_fd : file, gzipped file, pipe or opened file descriptor.
    """
    try:
        binary = fd.read(2).decode()
        if binary == '\0B' :
            mat = _read_mat_binary(fd)
        else:
            assert(binary == ' ['), binary
            mat = _read_mat_ascii(fd)
    finally:
        pass
    return mat

def read_ark(ark_ext_file_path: str):
    ark_file_path, location = ark_ext_file_path.split(':')
    with open(ark_file_path, 'rb') as f:
        f.seek(int(location))
        mat = read_mat(f)
        return mat 
    
def read_mat_hdf5(hdf5_file_path: str, utterance_id: str):
    with h5py.File(hdf5_file_path, 'r') as hf:
        if utterance_id not in hf:
            raise ValueError(f"utterance_id {utterance_id} not found in the HDF5 file")
        return hf[utterance_id][:]
    

def read_melspectrogram_from_batch(batch, max_length=None):
    # this function optimize reading by minimizing file access operations

    # read all melspectrogram paths and utterance ids
    file2utterance_id = {} 
    for item in batch:
        melspec_path, utterance_id = item['melspec_path'], item['utterance_id']
        if melspec_path not in file2utterance_id:
            file2utterance_id[melspec_path] = []
        file2utterance_id[melspec_path].append(utterance_id)
   
    # read melspectrogram features
    melspectrogram_features = []
    for melspec_path in file2utterance_id:
        with h5py.File(melspec_path, 'r') as hf:
            for utterance_id in file2utterance_id[melspec_path]:
                if utterance_id not in hf:
                    raise ValueError(f"utterance_id {utterance_id} not found in the HDF5 file {melspec_path}!")
                mel = hf[utterance_id][:]
                if max_length is not None:
                    mel = mel[:, :max_length]
                melspectrogram_features.append(mel)


    return melspectrogram_features


def log(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")