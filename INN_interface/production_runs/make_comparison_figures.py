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
    "blade.ps.layer_thickness_param",
    # "rotorse.stall_check.no_stall_constraint",
    # "blade.compute_reynolds.Re",
    # "rotorse.rp.powercurve.ax_induct_regII",
]
nice_data_names = [
    "Twist, deg",
    "Chord, m",
    "L/D",
    "Thickness/chord",
    "Sparcap thickness, m",
    # "rotorse.stall_check.no_stall_constraint",
    # "blade.compute_reynolds.Re",
    # "rotorse.rp.powercurve.ax_induct_regII",
]
airfoil_indices = [19, 24, 28]

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_data = len(data_names)
n_indices = len(airfoil_indices)
    
f, axarr = plt.subplots(n_data, 1, figsize=(6, 1.5*n_data), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, data_name in enumerate(data_names):
        if "L/D" in data_name:
            subdata = data["rotorse.ccblade.cl"][-1] / data["rotorse.ccblade.cd"][-1]
        elif 'layer_thickness' in data_name:
            subdata = data[data_name][-1, 3]
        else:
            subdata = data[data_name][-1]
        x = np.linspace(0., 1., len(subdata))
        y = subdata
        axarr[jdx].plot(x, y, label=labels[idx], clip_on=False)
        
        if "theta" in data_name and idx==0:
            x = data["blade.opt_var.s_opt_twist"][-1][2:]
            y = data["blade.opt_var.twist_opt"][-1][2:]
            axarr[jdx].scatter(x, y, s=20)
        
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(nice_data_names[jdx].split('.')[-1], rotation=0, ha='left', labelpad=112)

f.align_ylabels(axarr)
axarr[0].legend(loc='best', fontsize=10)    
axarr[-1].set_xlabel('Nondimensional blade span')
plt.savefig('span_quantities.pdf')

f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, idx_to_plot in enumerate(airfoil_indices):
        xy = data['blade.run_inn_af.coord_xy_interp'][-1, :, :, :]
        axarr[jdx].plot(xy[idx_to_plot, :, 0], xy[idx_to_plot, :, 1], label=labels[idx], clip_on=False)
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_aspect('equal', 'box')
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(f'AF index: {idx_to_plot}')

axarr[0].legend(loc='best', fontsize=8)
plt.savefig('airfoils.pdf')