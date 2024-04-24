import numpy as np
from sklearn.preprocessing import minmax_scale


def __rms(x):
    return np.sqrt(np.mean(x**2, axis=-1))


def __f(data):
    rms_values = []
    for j in range(data.shape[0]):
        rms_values.append(__rms(data[j]))
    return rms_values


def reshape_rms(rms_values):
    nsamples, nx, ny = rms_values.shape
    return rms_values.reshape((nsamples,nx*ny))


def generate_rms(dt, frame_length=None, hop_length=None):
    if frame_length is None and hop_length is not None:
        raise ValueError("hop_length cannot be set if frame_length isn't")
    
    data = dt[2]
    rms_values = []
    nb_datapoints = data.shape[0]
    nb_mics = data.shape[1]
    sample_length = data.shape[2]

    if frame_length is None and hop_length is None:
        for i in range(nb_datapoints):
            rms_values.append(__f(data[i, :, :]))
            print(i)
        return np.asarray(rms_values)
        # return np.asarray([ [rms(data[i,j]) for j in range(nb_mics)] for i in range(nb_datapoints)])
    if hop_length is None:
        hop_length = frame_length // 4

    for i in range(nb_datapoints):
        rms_values.append([])
        for j in range(nb_mics):
            rms_values[i].append([ __rms(data[i,j][k : k + frame_length ]) for k in range(0, sample_length, hop_length)])
        print(i)

    return reshape_rms(np.asarray(rms_values))


def normalize_rms(rms_values):
    rms_normed = np.clip(minmax_scale(rms_values, axis=1), 0,1)
    return rms_normed
