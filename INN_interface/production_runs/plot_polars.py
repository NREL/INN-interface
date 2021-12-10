import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


case_names = ['00']
labels = ['WISDEM', 'INN-WISDEM']
data_names = [
    "rotorse.rp.powercurve.L_D",
    # "blade.run_inn_af.cl_interp",
    # "blade.run_inn_af.cd_interp",
]
airfoil_indices = [19, 24, 28]

all_data, optimization_logs = load_cases(case_names)
    
f, ax = plt.subplots(len(data_names) + 1, 3, figsize=(len(case_names) * 4, 6), constrained_layout=True)

for i, idx in enumerate(airfoil_indices):
    for j, data_name in enumerate(data_names):
        for k, case_name in enumerate(case_names):
            s = all_data[k]["blade.outer_shape_bem.s"][-1]
            ax[j, i].plot(s, all_data[k][data_name][-1])
            print(s)
            print(all_data[k][data_name][-1])
            exit()
            
# plt.tight_layout()

plt.show()