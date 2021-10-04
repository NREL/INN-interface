import os
import numpy as np
import wisdem.inputs as sch

try:
    from INN_interface.INN import INN
    from INN_interface import utils
except:
    raise Exception("The INN framework for airfoil design is activated, but not installed correctly")
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from wisdem.ccblade.Polar import Polar
from wisdem.rotorse.geometry_tools.geometry import AirfoilShape, remap2grid, trailing_edge_smoothing

from INN_interface.cst import AirfoilShape as AirfoilShape_cst
from INN_interface.cst import CSTAirfoil
plt.rcParams["font.size"] = "16"
plt.rcParams["lines.linewidth"] = "3"



mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
geom_yaml_filename = os.path.dirname(mydir) + os.sep + "BAR_USC.yaml"
wt_init = sch.load_geometry_yaml(geom_yaml_filename)


airfoils = wt_init['airfoils']

individual_plots = False
Re = 9.0e6
fontsize = 16
alpha = np.arange(-4, 20.25, 0.25)

if not individual_plots:
    fig = plt.figure(figsize=(8, 6))


for i, airfoil in enumerate(airfoils):
    print(airfoil['name'])
    
    if 'circular' in airfoil['name'] or '500' in airfoil['name']:
        continue
        
    Re = airfoil['polars'][0]['re']
    
    inn = INN()

    x, y = airfoil['coordinates']['x'], airfoil['coordinates']['y']

    points = np.column_stack((x, y))
    # Check that airfoil points are declared from the TE suction side to TE pressure side
    idx_le = np.argmin(points[:, 0])
    if np.mean(points[:idx_le, 1]) > 0.0:
        points = np.flip(points, axis=0)

    # Remap points using class AirfoilShape
    af = AirfoilShape(points=points)
    af.redistribute(200, even=False, dLE=True)
    s = af.s
    af_points = af.points

    yaml_airfoil = AirfoilShape_cst(xco=af_points[:, 0], yco=af_points[:, 1])
    cst_new = CSTAirfoil(yaml_airfoil)
    cst = np.concatenate((cst_new.cst, [yaml_airfoil.te_lower], [yaml_airfoil.te_upper]), axis=0)

    cd_new, cl_new = inn.generate_polars(cst, Re, alpha=alpha)
    cd_new = np.squeeze(cd_new)
    cl_new = np.squeeze(cl_new)
    
    inn_polar = Polar(Re, alpha, cl_new, cd_new, np.zeros_like(cl_new))
    
    cl_inn = np.interp(alpha, inn_polar.alpha, inn_polar.cl)
    cd_inn = np.interp(alpha, inn_polar.alpha, inn_polar.cd)

    cl = airfoil['polars'][0]['c_l']
    cd = airfoil['polars'][0]['c_d']
    
    cl_yaml = np.interp(alpha, np.degrees(cl['grid']), cl['values'])
    cd_yaml = np.interp(alpha, np.degrees(cd['grid']), cd['values'])
    
    if individual_plots:

        f, ax = plt.subplots(4, 1, figsize=(5.3, 10))

        ax[0].plot(alpha, cl_inn, label="INN")
        ax[0].plot(alpha, cl_yaml, label="xfoil")
        # ax[0].plot(alpha * 180. / np.pi, cl_interp_new, label="xfoil")
        ax[0].grid(color=[0.8, 0.8, 0.8], linestyle="--")
        ax[0].legend()
        ax[0].set_ylabel("CL (-)", fontweight="bold", fontsize=fontsize)
        ax[0].set_title(f"{airfoil['name']} at {Re:.0f} Re", fontweight="bold", fontsize=fontsize)
        ax[0].set_ylim(-1.0, 2.5)
        ax[0].set_xlim(left=-4, right=20)

        ax[1].semilogy(alpha, cd_inn, label="INN")
        ax[1].semilogy(alpha, cd_yaml, label="xfoil")
        # ax[1].semilogy(alpha * 180. / np.pi, cd_interp_new, label="xfoil")
        ax[1].grid(color=[0.8, 0.8, 0.8], linestyle="--")
        ax[1].set_ylabel("CD (-)", fontweight="bold", fontsize=fontsize)
        ax[1].set_ylim(0.005, 0.2)
        ax[1].set_xlim(left=-4, right=20)

        ax[2].plot(alpha, cl_inn / cd_inn, label="INN")
        ax[2].plot(
            alpha,
            cl_yaml/cd_yaml,
            label="xfoil",
        )
        # ax[2].plot(alpha * 180. / np.pi, cl_interp_new / cd_interp_new, label="xfoil")
        ax[2].grid(color=[0.8, 0.8, 0.8], linestyle="--")
        ax[2].set_ylabel("CL/CD (-)", fontweight="bold", fontsize=fontsize)
        ax[2].set_xlabel("Angle of Attack (deg)", fontweight="bold", fontsize=fontsize)
        ax[2].set_xlim(left=-4, right=20)
        ax[2].set_ylim(top=170, bottom=-40)

        cst_xy = af_points

        ax[3].plot(cst_xy[:, 0], cst_xy[:, 1], label="INN")
        ax[3].plot(x, y, label="xfoil")
        ax[3].grid(color=[0.8, 0.8, 0.8], linestyle="--")
        ax[3].set_ylabel("y-coord", fontweight="bold", fontsize=fontsize)
        ax[3].set_xlabel("x-coord", fontweight="bold", fontsize=fontsize)
        ax[3].set_xlim(left=0.0, right=1.0)
        ax[3].set_ylim(top=0.2, bottom=-0.2)

        # plt.tight_layout()

        plt.savefig(f"airfoil_comparison_{i}.png", bbox_inches='tight', dpi=900)
        plt.close()
        
    else:
        ax = plt.gca()
        
        ax.plot(alpha, cl_inn / cd_inn, label=f"{airfoil['name']} at {Re:.0f} Re")
        # ax.plot(
        #     alpha,
        #     cl_yaml/cd_yaml,
        #     label=f"{airfoil['name']} at {Re:.0f} Re",
        # )
        ax.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        ax.set_ylabel("CL/CD (-)", fontweight="bold", fontsize=fontsize)
        ax.set_xlabel("Angle of Attack (deg)", fontweight="bold", fontsize=fontsize)
        ax.set_xlim(left=-4, right=20)
        ax.set_ylim(top=200, bottom=-40)
        

if not individual_plots:
    plt.legend()
    plt.tight_layout()
    plt.savefig('INN_different_Re.png', dpi=900)