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
optimization_logs = glob.glob(os.path.join(output_path, "log_opt.sql*"))

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
    
print(all_data)
        
import matplotlib.pyplot as plt

x_key = 'inn_af.L_D_opt'
y_key = 'rotorse.rp.powercurve.Cp_regII'

plt.plot(all_data[x_key][:, 2], all_data[y_key])
plt.xlabel(x_key)
plt.ylabel(y_key)

plt.show()
        