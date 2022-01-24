import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases
from INN_interface.production_runs.final_cases import case_names, labels


colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 5
data_names = [
    "rotorse.theta",
    "blade.pa.chord_param",
    "L/D",
    't/c',
    # "airfoil_thickness",
    "blade.ps.layer_thickness_param",
    # "rotorse.stall_check.no_stall_constraint",
    # "blade.compute_reynolds.Re",
    # "rotorse.rp.powercurve.ax_induct_regII",
    # "rotorse.rs.aero_gust.loads_Px",
    # "rotorse.rs.aero_gust.loads_Py",   
]
nice_data_names = [
    "Twist, deg",
    "Chord, m",
    "L/D",
    "Thickness/chord",
    # "Blade thickness, m",
    "Sparcap thickness, m",
    # "rotorse.stall_check.no_stall_constraint",
    # "blade.compute_reynolds.Re",
    # "rotorse.rp.powercurve.ax_induct_regII",
    # "gust Px",
    # "gust Py",
]
airfoil_indices = [19, 24, 28]
idx_case = -1

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_data = len(data_names)
n_indices = len(airfoil_indices)
    
f, axarr = plt.subplots(n_data, 1, figsize=(6, 1.2*n_data), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, data_name in enumerate(data_names):
        if "L/D" in data_name:
            subdata = data["rotorse.ccblade.cl"][idx_case] / data["rotorse.ccblade.cd"][idx_case]
        elif 'layer_thickness' in data_name:
            subdata = data[data_name][idx_case, 3]
        elif 'airfoil_thickness' in data_name:
            subdata = data['t/c'][idx_case] * data['blade.outer_shape_bem.chord'][idx_case]
            
        else:
            subdata = data[data_name][idx_case]
            
        x = np.linspace(0., 1., len(subdata))
        y = subdata
        axarr[jdx].plot(x, y, label=labels[idx], clip_on=False)
        
        if "theta" in data_name and idx==0:
            x = data["blade.opt_var.s_opt_twist"][idx_case][2:]
            y = data["blade.opt_var.twist_opt"][idx_case][2:]
            axarr[jdx].scatter(x, y, s=20, clip_on=False)
            
            axarr[jdx].annotate(labels[0], xy=(0.4, .55), fontsize=10,
                va="center", xycoords='axes fraction', ha='left', color=colors[0])
            axarr[jdx].annotate(labels[1], xy=(0.03, 0.12), fontsize=10,
                va="center", xycoords='axes fraction', ha='left', color=colors[1])
            
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(nice_data_names[jdx], rotation=0, ha='left', labelpad=112)

f.align_ylabels(axarr)  
axarr[-1].set_xlabel('Nondimensional blade span')
plt.savefig('span_quantities.pdf')

f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, idx_to_plot in enumerate(airfoil_indices):
        xy = data['blade.run_inn_af.coord_xy_interp'][idx_case, :, :, :]
        axarr[jdx].plot(xy[idx_to_plot, :, 0], xy[idx_to_plot, :, 1], label=labels[idx], clip_on=False)
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_aspect('equal', 'box')
        axarr[jdx].set_xlim([0., 1.])
        axarr[jdx].set_ylabel(f'AF index: {idx_to_plot}')

axarr[0].legend(loc='best', fontsize=8)
plt.savefig('airfoils.pdf')