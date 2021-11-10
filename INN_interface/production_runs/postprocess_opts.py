import glob
import os
import sys
import time

import numpy as np

import openmdao.api as om

from INN_interface.production_runs.postprocessing_tools import load_cases

case_names = ['00', '12']

all_data, optimization_logs = load_cases(case_names)
    
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
    "delta LCOE" : "financese.lcoe",
    "blade cost" : "tcc.blade_cost",
    "AEP" : "rotorse.rp.AEP",
}

limits = {
    "L/D" : [70., 130.],
    "CD" : [0.0, 0.08],
    "stall_m" : [-0.0, 0.4],
    "twist" : [-0.1, 0.3],
    "chord" : [2.0, 6.0],
    "spar_cap" : [0., 0.15],
    "LCOE" : [0.086, 0.090],
    "delta LCOE" : [-0.05, 0.01],
}

n_keys = len(keys_to_plot)

fig, axarr = plt.subplots(n_keys, n_cases, figsize=(30, 12))
for idx, data in enumerate(all_data):

    for jdx, (key, dat_key) in enumerate(keys_to_plot.items()):
        try:
            if 'delta' in key:
                data_to_plot = (data[dat_key] - data[dat_key][0]) / data[dat_key][0]
                iters = range(len(data[dat_key]))
                axarr[jdx, idx].plot(iters, data_to_plot)
            else:
                iters = range(len(data[dat_key]))
                data_to_plot = data[dat_key]
                axarr[jdx, idx].plot(iters, data_to_plot)
        except KeyError as e:
            pass
        axarr[jdx, idx].set_ylabel(key)
        
        if key in limits:
            axarr[jdx, idx].set_ylim(limits[key][0], limits[key][1])
    
    case_name = optimization_logs[idx].split('/')[-3:-2][0]
    axarr[0, idx].set_title(case_name)        
    axarr[-1, idx].set_xlabel('iteration')
    
plt.tight_layout()

plt.show()
# plt.savefig('opt_history.pdf')
