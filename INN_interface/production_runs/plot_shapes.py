import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots


case_names = ['00', '12']
airfoil_indices = [19, 24, 29]

optimization_logs = []

# Simply gather all of the sql files
run_dir = os.path.dirname(os.path.realpath(__file__))
for subdir, dirs, files in os.walk(run_dir):
    for file in files:
        if 'sql' in file:
            optimization_logs.append(os.path.join(subdir, file))

case_filenames = []
for log in optimization_logs:
    for case_name in case_names:
        if case_name in log:
            case_filenames.append(log)
            
optimization_logs = case_filenames
        
all_data = []
for idx, log in enumerate(optimization_logs):
    print(idx, log)
    data = {}
    cr = om.CaseReader(log)
    cases = cr.get_cases()
    
    for case in cases:
        for key in case.outputs.keys():
            if key not in data.keys():
                print(key)
                data[key] = []
            data[key].append(case.outputs[key])
            
    for key in data.keys():
        data[key] = np.array(data[key])
        
    all_data.append(data)
    
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

axarr[0].legend(loc='best')    
plt.show()