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

reynolds = [3.e6, 6.e6, 9.e6, 12.e6]
fontsize = 16
alpha = np.arange(-4, 20.25, 0.25)

f, axarr = plt.subplots((len(airfoils)-1) // 2, 2, sharex=True, figsize=(12, 12), dpi=300)
axarr = axarr.flatten()
for i, airfoil in enumerate(airfoils):
    print(airfoil['name'])
    
    for Re in reynolds:
        if 'circular' in airfoil['name']:
            continue
            
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
        try:
            cd_new, cl_new = inn.generate_polars(cst, Re, alpha=alpha)
        except:
            continue
        cd_new = np.squeeze(cd_new)
        cl_new = np.squeeze(cl_new)
        
        inn_polar = Polar(Re, alpha, cl_new, cd_new, np.zeros_like(cl_new))
        
        cl_inn = np.interp(alpha, inn_polar.alpha, inn_polar.cl)
        cd_inn = np.interp(alpha, inn_polar.alpha, inn_polar.cd)

        cl = airfoil['polars'][0]['c_l']
        cd = airfoil['polars'][0]['c_d']
        
        cl_yaml = np.interp(alpha, np.degrees(cl['grid']), cl['values'])
        cd_yaml = np.interp(alpha, np.degrees(cd['grid']), cd['values'])
        
        axarr[i-1].plot(alpha, cl_inn / cd_inn, label=f"{Re:.0f} Re")
        # axarr[i-1].plot(
        #     alpha,
        #     cl_yaml/cd_yaml,
        #     label=f"{airfoil['name']} at {Re:.0f} Re",
        # )
        axarr[i-1].grid(color=[0.8, 0.8, 0.8], linestyle="--")
        axarr[i-1].set_ylabel("CL/CD (-)", fontweight="bold", fontsize=fontsize)
        axarr[i-1].set_xlabel("Angle of Attack (deg)", fontweight="bold", fontsize=fontsize)
        axarr[i-1].set_xlim(left=-4, right=20)
        axarr[i-1].set_ylim(top=150, bottom=-40)
        axarr[i-1].set_title(airfoil["name"])
            
plt.sca(axarr[1])
plt.legend()
plt.tight_layout()
plt.savefig(f"INN.png")