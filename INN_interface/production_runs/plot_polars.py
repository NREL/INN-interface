import glob
import os

import numpy as np

import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


case_names = ['05', '19']
case_labels = ['WISDEM', 'INN-WISDEM']
data_names = [
    "blade.run_inn_af.cl_interp",
    "blade.run_inn_af.cd_interp",
    "l/d",
    "shape",
]
airfoil_indices = [20, 25, 28]
airfoil_labels = [.4483, .6897, .8621]
n_aoa = 200
opt_idx = -1


aoa = np.unique(
    np.hstack(
        [
            np.linspace(
                -np.pi, -np.pi / 6.0, int(n_aoa / 4.0 + 1)
            ),
            np.linspace(
                -np.pi / 6.0,
                np.pi / 6.0,
                int(n_aoa / 2.0),
            ),
            np.linspace(
                np.pi / 6.0, np.pi, int(n_aoa / 4.0 + 1)
            ),
        ]
    )
)
aoa = np.rad2deg(aoa)

all_data, optimization_logs = load_cases(case_names)
    
f, axarr = plt.subplots(len(data_names), 3, figsize=(10, 6), constrained_layout=True)

for i, idx in enumerate(airfoil_indices):
    for j, data_name in enumerate(data_names):
        for k, case_name in enumerate(case_names):
            ax = axarr[j, i]
            
            if 'l/d' in data_name:
                af = all_data[k]["blade.run_inn_af.cl_interp"][opt_idx][idx, :, 0, 0] / all_data[k]["blade.run_inn_af.cd_interp"][opt_idx][idx, :, 0, 0]
            elif 'shape' in data_name:
                xy = all_data[k]['blade.run_inn_af.coord_xy_interp'][opt_idx, :, :, :]
            else:
                af = all_data[k][data_name][opt_idx][idx, :, 0, 0]
                
            if 'shape' in data_name:
                ax.plot(xy[idx, :, 0], xy[idx, :, 1], clip_on=False)
            else:
                ax.plot(aoa, af, label=case_labels[k])
                ax.set_xlim(left=-4, right=20)
                ax.set_xlabel("Angle of Attack (deg)")
            
            if 'cl' in data_name:
                ax.set_ylim(-0.2, 2.75)
                if i==0:
                    ax.legend(fontsize=8)
                    ax.set_ylabel("CL")
                ax.set_title("Span Location {:2.2%}".format(airfoil_labels[i]))
            elif 'cd' in data_name:
                ax.set_yscale('log')
                ax.set_ylim(0.005, 0.1)
                if i==0:
                    ax.set_ylabel("CD")
            elif 'l/d' in data_name:
                ax.set_ylim(top=130, bottom=-20)
                if i==0:
                    ax.set_ylabel("L/D")
            elif 'shape' in data_name:
                ax.set_xlim(left=0.0, right=1.0)
                ax.set_ylim(top=0.2, bottom=-0.2)
                ax.set_axis_off()
            
plt.tight_layout()

plt.show()