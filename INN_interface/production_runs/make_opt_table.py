import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


data_names = [
    "tcc.blade_cost",
    "tcc.rotor_cost",
    "tcc.turbine_cost",
    "rotorse.rp.AEP",
    "financese.lcoe",
    ]
    
case_names = ['05', '05', '19']
indices = [0, -1, -1]
labels = ["Initial", "WISDEM", "INN-WISDEM"]

n_cases = len(case_names)
n_data = len(data_names)

all_data, optimization_logs = load_cases(case_names)


all_lines = []
for idx, (plot_idx, case_name) in enumerate(zip(indices, case_names)):
    data = all_data[idx]
    line = []
    for jdx, data_name in enumerate(data_names):
        line.append(str(data[data_name][plot_idx][0]))
    all_lines.append(line)

trimmed_data = [data_name.split('.')[-1] for data_name in data_names]
print(' & ' + ' & '.join(trimmed_data), ' \\\\')
for i, line in enumerate(all_lines):
    print(f"{labels[i]} & " + ' & '.join(list(line)) + ' \\\\')