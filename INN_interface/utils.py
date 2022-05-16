import os
import h5py
import numpy as np

def norm_data(data, scale_factor=None, scale_type='standard'):
    # Function to normalize the input data
    # If scale_type is 'standard', then scale_factor contains [mean, std. dev.]
    # If scale_type is 'minmax', then scale_factor contains [min, max]
    # Computes appropriate scale_factors from data if not provided
    if scale_type == 'standard':
        if scale_factor is None:
            mu, sig = np.mean(data, axis=0), np.std(data, axis=0)
            sig[sig == 0.] = 1.
            scale_factor = [mu, sig, 'standard']
        data = (data - scale_factor[0])/scale_factor[1]

    elif scale_type == 'minmax':
        if scale_factor is None:
            scale_factor = [np.min(data, axis=0), np.max(data, axis=0), 'minmax']
            tf = (scale_factor[0] == scale_factor[1])
            for i, tf_i in enumerate(tf):
                if tf_i:
                    if scale_factor[0][i] != 0.:
                        scale_factor[0][i], scale_factor[1][i] = 0., scale_factor[1][i]/2
                    else:
                        scale_factor[0][i], scale_factor[1][i] = -1., 1.
        data = 2.*(data - scale_factor[0])/(scale_factor[1] - scale_factor[0]) - 1.

    else:
        assert False, 'Scaling type invalid.'

    return data, scale_factor

def unnorm_data(data, scale_factor, scale_type='standard'):
    # Function to unnormalize the input data to physical space
    # If scale_type is 'standard', then scale_factor contains [mean, std. dev.]
    # If scale_type is 'minmax', then scale_factor contains [min, max]
    if scale_type == 'standard':
        data = scale_factor[1]*data + scale_factor[0]

    elif scale_type == 'minmax':
        data = 0.5*(scale_factor[1] - scale_factor[0])*(data + 1.) + scale_factor[0]

    else:
        assert False, 'Scaling type invalid.'
    
    return data

def save_scale_factors(af_name, scale_factors):
    # Function to write scale_factors to h5 file 
    this_directory = os.path.abspath(os.path.dirname(__file__))
    input_file_sf = os.path.join(this_directory, "model/" + af_name + '/scale_factors.h5')
    with h5py.File(input_file_sf, 'w') as f:
        for key in scale_factors:
            f_key = f.create_group(key)
            if scale_factors[key][2] == 'minmax':
                f_key.create_dataset('type', data=scale_factors[key][2])
                f_key.create_dataset('min', data=scale_factors[key][0])
                f_key.create_dataset('max', data=scale_factors[key][1])
            elif scale_factors[key][2] == 'standard':
                f_key.create_dataset('type', data=scale_factors[key][2])
                f_key.create_dataset('mu', data=scale_factors[key][0])
                f_key.create_dataset('sigma', data=scale_factors[key][1])
            else:
                assert False, 'Bad scale type'

def load_scale_factors(filepath):
    # Function to load previously saved scale_factors
    scale_factors = {}
    this_directory = os.path.abspath(os.path.dirname(__file__))
    input_file_sf = os.path.join(this_directory, filepath+'/scale_factors.h5')
    with h5py.File(input_file_sf, 'r') as f:
        for key in f:
            f_key = f[key]
            if f_key['type'][()] == 'minmax':
                scale_factors[key] = [f_key['min'][()], f_key['max'][()], 'minmax']
            else:
                scale_factors[key] = [f_key['mu'][()], f_key['sigma'][()], 'standard']

    return scale_factors



