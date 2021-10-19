import numpy as np
from INN_interface.INN_pga import INN
from INN_interface import utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

cd = 0.015
clcd = 83.
stall_margin = 5.
thickness = 0.25
Re = 9000000.

inn = INN()

XY_inn, alpha, Re_inn, z = inn.inverse_design(cd, clcd, stall_margin, thickness, Re, z=1, N=10, 
                                              process_samples=True, return_z=True, data_format='XY')

print('Optimal z for requested values: {}'.format(z[0, :]))

alpha = np.arange(-4, 20, 0.25)
cd, cl = inn.generate_polars(XY_inn, Re_inn, alpha=alpha, 
                             KL_basis=True, data_format='XY')

with PdfPages('af_design_shapes.pdf') as pfpgs:
    fig = plt.figure()
    for i, xy_i in enumerate(XY_inn): 
        plt.plot(xy_i[:, 0], xy_i[:, 1], 'k--', linewidth=0.5)
    plt.grid()
    plt.xlabel('x/c')    
    plt.ylabel('y/c')
    plt.tight_layout()
    pfpgs.savefig()
    
    fig,ax = plt.subplots(1,3)
    for i, xy_i in enumerate(XY_inn):
        ax[0].plot(alpha, cl[i], 'k--', linewidth=0.1)
        ax[1].plot(alpha, cd[i], 'k--', linewidth=0.1)
        ax[2].plot(alpha, cl[i]/cd[i], 'k--', linewidth=0.1)
    ax[0].set_xlabel('$\alpha$')
    ax[1].set_xlabel('$\alpha$')
    ax[2].set_xlabel('$\alpha$')
    ax[0].set_xlabel('$C_l$')
    ax[1].set_xlabel('$C_d$')
    ax[2].set_xlabel('$L/D$')
    plt.tight_layout()
    pfpgs.savefig()
    plt.close(fig)
        
