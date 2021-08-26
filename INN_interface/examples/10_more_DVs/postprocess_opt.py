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
import niceplots

# Simply gather all of the sql files
run_dir = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(run_dir, "outputs")
optimization_logs = glob.glob(os.path.join(output_path, "*.sql*"))

print(optimization_logs)

all_data = {}
for log in optimization_logs:
    cr = om.CaseReader(log)
    cases = cr.get_cases()
    
    for case in cases:
        for key in case.outputs.keys():
            if key not in all_data.keys():
                all_data[key] = []
            all_data[key].append(case.outputs[key])
            
            
for key in all_data.keys():
    all_data[key] = np.array(all_data[key])
    
import matplotlib.pyplot as plt

l_d = 'inn_af.L_D_opt'
c_d = 'inn_af.c_d_opt'
twist = 'blade.opt_var.twist_opt'
cp = 'rotorse.rp.powercurve.Cp_regII'

cutoff_idx = 150

l_d = all_data[l_d][:cutoff_idx, 2:]
c_d = all_data[c_d][:cutoff_idx, 2:]
twist = all_data[twist][:cutoff_idx, 2:]
cp = all_data[cp][:cutoff_idx]
iters = range(len(cp))

print(twist[0, :])

labels = ['L/D', 'Cd', 'Twist, deg', 'Cp']
ticks = np.arange(0, cutoff_idx/30 + 1) * 30
yticks = [[60, 90, 120], [0.014, 0.017, 0.020], [-4, 2, 8], [.44, .45, .46]]

fig, ax = plt.subplots(4, 1)
ax[0].plot(iters, l_d)
ax[1].plot(iters, c_d)
ax[2].plot(iters, np.rad2deg(twist))
ax[-1].plot(iters, cp, color='k')
ax[-1].set_xlabel('Optimization iteration')

for idx, label in enumerate(labels):
    ax[idx].set_ylabel(label, rotation=0, labelpad=30, ha='left')
    ax[idx].set_xticks(ticks)
    ax[idx].set_yticks(yticks[idx])
    
    
fig.align_ylabels(ax)

for subax in ax:
    niceplots.adjust_spines(subax)

plt.tight_layout()

perc_cp = (cp[-1] - cp[0]) / cp[0] * 100.
ax[-1].annotate(f'{perc_cp[0]:.2f}% increase in Cp', (.7, 0.5), fontsize=10, xycoords='axes fraction', annotation_clip=False)

plt.savefig('opt_history.png', dpi=300)
