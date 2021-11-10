import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots


case_names = ['03', '13']
data_names = [
    "tcc.blade_cost",
    "tcc.rotor_cost",
    "tcc.turbine_cost",
    "rotorse.rp.AEP",
    "financese.lcoe",
    ]

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
                # print(key)
                data[key] = []
            data[key].append(case.outputs[key])
            
    for key in data.keys():
        data[key] = np.array(data[key])
        
    all_data.append(data)
    
n_cases = len(optimization_logs)
n_indices = len(data_names)
    
f, axarr = plt.subplots(n_indices, 1, figsize=(4, 1.5*n_indices), constrained_layout=True)

for idx, data in enumerate(all_data):
    for jdx, data_name in enumerate(data_names):
        subdata = data[data_name]
        x = range(len(subdata))
        y = subdata
        axarr[jdx].plot(x, y, label=case_names[idx])
        niceplots.adjust_spines(axarr[jdx])
        axarr[jdx].set_ylabel(data_name.split('.')[-1])

axarr[0].legend(loc='best')    
axarr[-1].set_xlabel('Nondimensional blade span')
plt.show()

