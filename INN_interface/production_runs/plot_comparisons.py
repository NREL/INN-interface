import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


case_names = ['03', '21']
data_names = ["inn_af.L_D_opt", "inn_af.c_d_opt", "inn_af.r_thick_opt", "blade.run_inn_af.aoa_inn"]

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_indices = len(data_names)
    
f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, data_name in enumerate(data_names):
        subdata = data[data_name][-1]
        x = np.linspace(0., 1., len(subdata))
        y = subdata
        axarr[jdx].plot(x, y, label=case_names[idx])
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(data_name.split('.')[-1])

axarr[0].legend(loc='best')    
axarr[-1].set_xlabel('Nondimensional blade span')
plt.show()

