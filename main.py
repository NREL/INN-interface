import numpy as np
from INN import INN
from utils import *
import matplotlib.pyplot as plt

cd = 0.015
clcd = 83.
stall_margin = 5.
thickness = 0.25
Re = 9000000.

inn = INN()
cst, alpha = inn.inverse_design(cd, clcd, stall_margin, thickness, Re, 
                                N=100, process_samples=True)
# cst is a numpy array of size (N x 20) where the columns are:
#    - 9 upper surface CST params
#    - 9 lower surface CST params
#    - upper trailing edge thickness
#    - lower trailing edge thickness

alpha = np.arange(-4, 20, 0.25)
cd, cl = inn.generate_polars(cst, Re, alpha=alpha)
