import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


data_names = [
    "tcc.blade_cost",
    # "tcc.rotor_cost",
    # "tcc.turbine_cost",
    "rotorse.rp.AEP",
    "financese.lcoe",
    ]
nice_labels = [
    "Blade cost, $M",
    # "Rotor cost, $M",
    # "Turbine cost, $M",
    "AEP, GWh",
    "LCOE, $/MWh",
    ]
    
case_names = ['05', '19']
nice_case_names = ['WISDEM', 'INN-WISDEM']

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_indices = len(data_names)
    
f, axarr = plt.subplots(n_indices, 1, figsize=(6, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, data_name in enumerate(data_names):
        subdata = data[data_name]
        x = range(len(subdata))
        y = subdata
        
        if 'blade_cost' in data_name:
            y /= 1.e6
        elif 'AEP' in data_name:
            y /= 1.e6
        elif 'lcoe' in data_name:
            y *= 1.e3
            
        axarr[jdx].plot(x, y, label=nice_case_names[idx], zorder=-idx)
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_ylabel(nice_labels[jdx], rotation=0, ha='left', labelpad=90)
        
axarr[0].legend(loc='best')    
axarr[-1].set_xlabel('Optimization iterations')
plt.savefig('iterations.pdf')

