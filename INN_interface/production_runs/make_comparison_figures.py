import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


case_names = ['05', '19']
labels = ['WISDEM', 'INN-WISDEM']
data_names = [
    "rotorse.theta",
    "blade.pa.chord_param",
    "L/D",
    't/c',
    "blade.internal_structure_2d_fem.layer_thickness",
]
airfoil_indices = [19, 24, 28]

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_indices = len(data_names)
    
f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, data_name in enumerate(data_names):
        if "L/D" in data_name:
            subdata = data["rotorse.ccblade.cl"][-1] / data["rotorse.ccblade.cd"][-1]
        elif 'layer_thickness' in data_name:
            subdata = data[data_name][-1, 2]
        else:
            subdata = data[data_name][-1]
        x = np.linspace(0., 1., len(subdata))
        y = subdata
        axarr[jdx].plot(x, y, label=labels[idx])
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(data_name.split('.')[-1])

axarr[0].legend(loc='best', fontsize=10)    
axarr[-1].set_xlabel('Nondimensional blade span')
plt.savefig('span_quantities.png')

f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, idx_to_plot in enumerate(airfoil_indices):
        xy = data['blade.run_inn_af.coord_xy_interp'][-1, :, :, :]
        axarr[jdx].plot(xy[idx_to_plot, :, 0], xy[idx_to_plot, :, 1], label=labels[idx])
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_aspect('equal', 'box')
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(f'AF index: {idx_to_plot}')

axarr[0].legend(loc='best', fontsize=8)
plt.savefig('airfoils.png')