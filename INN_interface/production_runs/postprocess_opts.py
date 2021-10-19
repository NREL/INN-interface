"""
Simple script to show how to grab all cases from a DOE run. User can then
postprocess or plot further.
"""

import glob
import os
import sys
import time

import numpy as np

import openmdao.api as om


optimization_logs = []

# Simply gather all of the sql files
run_dir = os.path.dirname(os.path.realpath(__file__))
for subdir, dirs, files in os.walk(run_dir):
    for file in files:
        if 'sql' in file:
            optimization_logs.append(os.path.join(subdir, file))

print(optimization_logs)

all_data = []
for idx, log in enumerate(optimization_logs):
    data = {}
    cr = om.CaseReader(log)
    cases = cr.get_cases()
    
    for case in cases:
        for key in case.outputs.keys():
            if key not in data.keys():
                data[key] = []
            data[key].append(case.outputs[key])
            
    for key in data.keys():
        data[key] = np.array(data[key])
        
    all_data.append(data)
    
# print(all_data)
# optimization_logs = optimization_logs[:2]

n_cases = len(optimization_logs)
    
import matplotlib.pyplot as plt

keys_to_plot = {
    "L/D" : "inn_af.L_D_opt",
    "CD" : "inn_af.c_d_opt",
    "stall_m" : "inn_af.stall_margin_opt",
    "twist" : "blade.opt_var.twist_opt",
    "chord" : "blade.opt_var.chord_opt",
    "spar_cap" : "blade.opt_var.spar_cap_ss_opt",
    "LCOE" : "financese.lcoe",
}

n_keys = len(keys_to_plot)

fig, axarr = plt.subplots(n_keys, n_cases, figsize=(30, 12))
for idx, data in enumerate(all_data):

    for jdx, (key, dat_key) in enumerate(keys_to_plot.items()):
        print(key, dat_key)
        try:
            iters = range(len(data[dat_key]))
            data_to_plot = data[dat_key]
            axarr[jdx, idx].plot(iters, data_to_plot)
        except KeyError:
            pass
        axarr[jdx, idx].set_ylabel(key)
    
    case_name = optimization_logs[idx].split('/')[-3:-2][0]
    axarr[0, idx].set_title(case_name)        
    axarr[-1, idx].set_xlabel('iteration')
    
plt.tight_layout()

plt.savefig('opt_history.pdf')
