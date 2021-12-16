import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

from INN_interface.production_runs.postprocessing_tools import load_cases


case_names = ['19']
airfoil_indices = [19, 24, 29]
data_names = [
    "financese.lcoe",
    "rotorse.rp.AEP",
    "tcc.blade_cost",
    ]
blade_data_names = [
    # "rotorse.theta",
    "rotorse.rp.powercurve.ax_induct_regII",
    "rotorse.rp.powercurve.L_D",
    't/c',
]
powercurve_names = [
    "rotorse.rp.powercurve.Cp_aero",
    "rotorse.rp.powercurve.Ct_aero",
    "rotorse.rp.powercurve.pitch",
]

all_data, optimization_logs = load_cases(case_names)
    
n_cases = len(optimization_logs)
n_indices = len(airfoil_indices)
n_data = len(data_names)

max_iterations = len(all_data[0]['financese.lcoe'])
# max_iterations = 350

from pathlib import Path
Path("animation").mkdir(parents=True, exist_ok=True)

for idx_animate in range(max_iterations):
    print(f'Animating frame {idx_animate+1}/{max_iterations}')
    f, axarr = plt.subplots(n_indices, 4, figsize=(15, 1.5*n_indices), constrained_layout=True)

    # Airfoil shapes
    for idx, data in enumerate(all_data):
        for jdx, idx_to_plot in enumerate(airfoil_indices):
            xy = data['blade.run_inn_af.coord_xy_interp'][0, :, :, :]
            axarr[jdx, 0].plot(xy[idx_to_plot, :, 0], xy[idx_to_plot, :, 1], color='gray')
            
            xy = data['blade.run_inn_af.coord_xy_interp'][idx_animate, :, :, :]
            axarr[jdx, 0].plot(xy[idx_to_plot, :, 0], xy[idx_to_plot, :, 1], label=case_names[idx])
            niceplots.adjust_spines(axarr[jdx, 0])
            axarr[jdx, 0].set_aspect('equal', 'box')
            axarr[jdx, 0].set_xlim([0., 1.])
            axarr[jdx, 0].set_ylabel(f'AF index: {idx_to_plot}')
            
    # Values across blade span
    for idx, data in enumerate(all_data):
        for jdx, data_name in enumerate(blade_data_names):
            subdata = data[data_name]

            y = subdata[0]
            x = np.linspace(0., 1., len(y))
            
            if 'aoa' in data_name:
                mask_value = 2.
                y[y > mask_value] = np.nan
            axarr[jdx, 1].plot(x, y, color='gray')
            
            y = subdata[idx_animate]
            x = np.linspace(0., 1., len(y))
            
            if 'aoa' in data_name:
                mask_value = 2.
                y[y > mask_value] = np.nan
            axarr[jdx, 1].plot(x, y, label=case_names[idx])
            
            niceplots.adjust_spines(axarr[jdx, 1])
            axarr[jdx, 1].set_xlim([0., 1.])
            axarr[jdx, 1].set_ylabel(data_name.split('.')[-1])
            
    # Values across powercurve
    for idx, data in enumerate(all_data):
        for jdx, data_name in enumerate(powercurve_names):
            subdata = data[data_name]

            y = subdata[0]
            x = data["rotorse.rp.powercurve.V"][0]
            axarr[jdx, 2].plot(x, y, color='gray')
            
            y = subdata[idx_animate]
            x = data["rotorse.rp.powercurve.V"][idx_animate]
            axarr[jdx, 2].plot(x, y, label=case_names[idx])
            
            niceplots.adjust_spines(axarr[jdx, 2])
            axarr[jdx, 2].set_xlim([3., 25.])
            axarr[jdx, 2].set_ylabel(data_name.split('.')[-1])
            
    # Optimization histories
    for idx, data in enumerate(all_data):
        for jdx, data_name in enumerate(data_names):
            subdata = data[data_name]
            x = range(len(subdata))
            y = subdata
                
            axarr[jdx, 3].scatter(x[idx_animate], y[idx_animate], color='k', zorder=100)
            axarr[jdx, 3].plot(x, y, label=case_names[idx])
            niceplots.adjust_spines(axarr[jdx, 3])
            axarr[jdx, 3].set_ylabel(data_name.split('.')[-1])
            
            percent_change = float((y[idx_animate] - y[0]) / y[0]) * 100.
            axarr[jdx, 3].annotate(f'{percent_change:.2f}% change', xy=(1.0, 0.5),
                va="center", xycoords='axes fraction', ha='right')

    axarr[-1, 1].set_xlabel('Nondimensional blade span')
    axarr[-1, 2].set_xlabel('Nondimensional velocity')
    axarr[-1, 3].set_xlabel('Optimization iterations')
    
    plt.savefig(f'animation/' + str(idx_animate).zfill(3) + '.png', dpi=300)
    
    plt.close()
    
last_num = max_iterations - 1  

for i in range(30):
    os.system(f'cp animation/' + str(last_num).zfill(3) + '.png animation/' + str(i + 1 + last_num).zfill(3) + '.png')

os.system(f"ffmpeg -r 15 -i animation/%3d.png -vcodec libx264 -b 5000k -y animation/movie.mp4")
os.system(f"ffmpeg -i animation/movie.mp4 -q:v 5 -y animation/output.wmv")