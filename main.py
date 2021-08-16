import numpy as np
from INN_interface.INN import INN
from INN_interface import utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

cd = 0.015
clcd = 83.
stall_margin = 5.
thickness = 0.25
Re = 9000000.

inn = INN()
af_csts, alpha = inn.inverse_design(cd, clcd, stall_margin, thickness, Re, 
                                    z=42, N=100, process_samples=True)
#
# cst is a numpy array of size (N x 20) where the columns are:
#    - 9 upper surface CST params
#    - 9 lower surface CST params
#    - upper trailing edge thickness
#    - lower trailing edge thickness

alpha = np.arange(-4, 20, 0.25)
cd, cl = inn.generate_polars(af_csts, Re, alpha=alpha)


with PdfPages('af_design_shapes.pdf') as pfpgs:
    fig = plt.figure()
    for af_cst in af_csts:
        x,y = utils.get_airfoil_shape(af_cst)       
        plt.plot(x, y, 'k--', linewidth=0.1)
    plt.grid()
    plt.xlabel('x/c')    
    plt.ylabel('y/c')
    plt.tight_layout()
    pfpgs.savefig()
    
    fig,ax = plt.subplots(1,3)
    for i, af_cst in enumerate(af_csts):
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
        
