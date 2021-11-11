import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


case_names = ['00', '12']
airfoil_indices = [19, 24, 29]

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_indices = len(airfoil_indices)
    
f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, idx_to_plot in enumerate(airfoil_indices):
        xy = data['blade.run_inn_af.coord_xy_interp'][-1, :, :, :]
        axarr[jdx].plot(xy[idx_to_plot, :, 0], xy[idx_to_plot, :, 1], label=case_names[idx])
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_aspect('equal', 'box')
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(f'AF index: {idx_to_plot}')

axarr[0].legend(loc='best')    
plt.show()