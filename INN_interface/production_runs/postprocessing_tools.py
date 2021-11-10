import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots


def load_cases(case_names=None):
    optimization_logs = []

    # Simply gather all of the sql files
    run_dir = os.path.dirname(os.path.realpath(__file__))
    for subdir, dirs, files in os.walk(run_dir):
        for file in files:
            if 'sql' in file:
                optimization_logs.append(os.path.join(subdir, file))

    
    if case_names is not None:
        case_filenames = []
        for log in optimization_logs:
            for case_name in case_names:
                if case_name in log:
                    case_filenames.append(log)
                    
        optimization_logs = case_filenames
            
    all_data = []
    for idx, log in enumerate(optimization_logs):
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
        
    return all_data, optimization_logs