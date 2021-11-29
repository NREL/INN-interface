import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


case_names = ['00', '12']
data_names = [
    "rotorse.theta",
    "blade.pa.chord_param",
    "l/d",
]
airfoil_indices = [19, 24, 29]

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_indices = len(data_names)
    
f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, data_name in enumerate(data_names):
        if "l/d" in data_name:
            subdata = data["rotorse.ccblade.cl"][-1] / data["rotorse.ccblade.cd"][-1]
        else:
            subdata = data[data_name][-1]
        x = np.linspace(0., 1., len(subdata))
        y = subdata
        axarr[jdx].plot(x, y, label=case_names[idx])
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(data_name.split('.')[-1])

axarr[0].legend(loc='best')    
axarr[-1].set_xlabel('Nondimensional blade span')
plt.savefig('span_quantities.png')

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
plt.savefig('airfoils.png')