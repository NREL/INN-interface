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

l_d = all_data[l_d][:, 2:]
c_d = all_data[c_d][:, 2:]
twist = all_data[twist][:, 2:]
cp = all_data[cp]
iters = range(len(cp))

fig, ax = plt.subplots(4, 1)
ax[0].plot(iters, l_d)
ax[1].plot(iters, c_d)
ax[2].plot(iters, twist)
ax[-1].plot(iters, cp)
ax[0].set_xlabel('iteration')
ax[-1].set_xlabel('iteration')
ax[0].set_ylabel('L/D')
ax[1].set_ylabel('cd')
ax[2].set_ylabel('twist')
ax[-1].set_ylabel('Cp')
plt.tight_layout()

plt.savefig('opt_history.png', dpi=300)
