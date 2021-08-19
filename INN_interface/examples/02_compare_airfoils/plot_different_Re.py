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

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
geom_yaml_filename = os.path.dirname(mydir) + os.sep + "BAR_USC.yaml"
wt_init = sch.load_geometry_yaml(geom_yaml_filename)


airfoils = wt_init['airfoils']

alpha = np.arange(-4, 20.25, 0.25)
Reynolds = np.linspace(3.e6, 12.e6, 4)

fig = plt.figure()


for i, airfoil in enumerate(airfoils):
    print(airfoil['name'])
    
    if '241' in airfoil['name']:
        for j, Re in enumerate(Reynolds):
            print(airfoil['name'], Re)
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
            
            ax = plt.gca()
            
            if j == 0:
                ax.plot(
                alpha,
                cl_yaml/cd_yaml,
                label=f"XFOIL at {airfoil['polars'][0]['re']:.0e} Re",
                color='k',
                )
            ax.plot(alpha, cl_inn / cd_inn, label=f"INN at {Re:.0e} Re")
            ax.grid(color=[0.8, 0.8, 0.8], linestyle="--")
            ax.set_ylabel("CL/CD (-)", fontweight="bold")
            ax.set_xlabel("Angle of Attack (deg)", fontweight="bold")
            ax.set_xlim(left=-4, right=20)
            ax.set_ylim(top=160, bottom=-20)
                

plt.title(airfoil['name'])
plt.legend()
plt.tight_layout()
plt.savefig('INN_reynolds_comparison.png')