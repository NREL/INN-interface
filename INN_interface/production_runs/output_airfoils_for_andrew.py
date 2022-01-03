import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import pickle

from INN_interface.production_runs.postprocessing_tools import load_cases


case_names = ['00', '05', '19']
labels = ['initial', 'WISDEM', 'INN-WISDEM']
data_names = [
    "blade.compute_reynolds.Re",
    "blade.pa.chord_param",
]
airfoil_indices = np.arange(30)

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_data = len(data_names)
n_indices = len(airfoil_indices)

output_data = {}

for idx, data in enumerate(all_data):
    output_data[labels[idx]] = {}
    output_data[labels[idx]]['coords'] = data['blade.run_inn_af.coord_xy_interp'][-1, :, :, :]
    output_data[labels[idx]]['reynolds'] = data["blade.compute_reynolds.Re"][-1, :]
    output_data[labels[idx]]['chord'] = data["blade.pa.chord_param"][-1, :]
    output_data[labels[idx]]['aoa'] = data["blade.run_inn_af.aoa_inn"][-1]
    

with open('outputted_airfoils.pkl', 'wb') as f:
    pickle.dump(output_data, f)
    
import pickle
with open('outputted_airfoils.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data)