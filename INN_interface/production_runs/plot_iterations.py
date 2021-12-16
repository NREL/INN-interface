import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


data_names = [
    "tcc.blade_cost",
    "tcc.rotor_cost",
    "tcc.turbine_cost",
    "rotorse.rp.AEP",
    "financese.lcoe",
    ]
    
case_names = ['05', '19']

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_indices = len(data_names)
    
f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, data_name in enumerate(data_names):
        subdata = data[data_name]
        x = range(len(subdata))
        y = subdata
        axarr[jdx].plot(x, y, label=case_names[idx])
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_ylabel(data_name.split('.')[-1])

axarr[0].legend(loc='best')    
axarr[-1].set_xlabel('Optimization iterations')
plt.savefig('iterations.pdf')

