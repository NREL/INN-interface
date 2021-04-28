# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

"""
Class Shape Transform (CST) utilities
"""

import numpy as np
import scipy.optimize as sopt
from scipy.special import comb
from scipy.interpolate import PchipInterpolator


class CST2D:
    """CST representation for 2-D shapes"""

    #: Mapping of the ``N1`` and ``N2`` coefficients for various shapes
    shape_class = {
        'airfoil' : (0.5, 1.0),
        'ellipse' : (0.5, 0.5),
        'biconvex': (1.0, 1.0),
        'sears_hack': (0.75, 0.75),
        'projectile': (0.75, 0.25),
        'cone' : (1.0, 0.001),
        'rectangle' : (0.001, 0.001)
    }

    def __init__(self, order=8):
        """
        Args:
            order (int): Bernstein polynomial order
        """
        #: Bernstein polynomial order of the CST parametrization
        self.order = order
        #: Polynomial coefficients
        self.kvals = comb(order, range(order + 1))
        # Assume default is airfoil
        self.n1, self.n2 = self.shape_class["airfoil"]

    def cls_fcn(self, xco):
        """Class function for a given psi = (x/c)

        Return:
            np.ndarray: Array containing class functions with length of xinp
        """
        return np.power(xco, self.n1) * np.power((1.0 - xco), self.n2)

    def shape_fcn(self, xinp):
        """Shape functions for a given psi = (x/c)

        Return:
            np.ndarray: ``[BP+1, len(xinp)]`` array of shape function
        """
        K = self.kvals
        N = self.order
        xco = np.asarray(xinp)

        stmp = np.empty((N+1, xco.shape[0]))
        for i in range(N + 1):
            stmp[i, :] = K[i] * np.power(xco, i) * np.power((1.0 - xco), (N-i))
        return stmp

    def cst_matrix(self, xinp):
        """Return product of C(psi) * S_i(psi)"""
        return (self.cls_fcn(xinp) * self.shape_fcn(xinp)).T

    def solve(self, xco, yco, yte=0.0):
        """Solve the least squares problem for a given shape

        Args:
            xco (np.ndarray): (x/c) coordinates locations
            yco (np.ndarray): (y/c) coordinate locations
            yte (double): Trailing edge thickness

        Return:
            np.ndarray: ``(BP+1)`` CST parameters
        """
        amat = self.cst_matrix(xco)
        bvec = yco - xco * yte
        out = sopt.lsq_linear(amat, bvec)
        return out



class CSTAirfoil(CST2D):
    """Concrete implementation of CST for airfoils"""

    def __init__(self, airfoil, order=8, shape_class='airfoil'):
        """
        Args:
            airfoil (AirfoilShape): Airfoil geometry information
            order (int): Polynomial order
            shape_class (string): Shape class
        """
        super().__init__(order)
        self.n1, self.n2 = self.shape_class[shape_class]
        #: Instance of :class:`~uaero_ml.aero.airfoil.AirfoilShape`
        self.airfoil = airfoil

    @classmethod
    def from_cst_parameters(cls, airfoil, cst_lower, cst_upper,
                            n1=0.5, n2=1.0):
        """
        Args:
            airfoil (AirfoilShape): Airfoil geometry information
            cst_lower: Lower surface CST coefficients
            cst_upper: Upper surface CST coefficients
        """
        order = np.size(cst_lower)-1
        self = CSTAirfoil(airfoil, order)
        self.n1 = n1
        self.n2 = n2
        self._cst_lower = cst_lower
        self._cst_upper = cst_upper
        self._cst = np.r_[self._cst_lower, self._cst_upper]
        return self

    def _compute_cst(self):
        """Compute CST coefficients for airfoil on demand"""
        af = self.airfoil
        out1 = self.solve(af.xupper, af.yupper, af.te_upper)
        self._cst_upper = out1.x
        out2 = self.solve(af.xlower, af.ylower, af.te_lower)
        self._cst_lower = out2.x
        self._cst = np.r_[self._cst_lower, self._cst_upper]
        return (out1, out2)

    @property
    def cst(self):
        """CST coefficients for the airfoil

        Returns an array of size (2*(BP+1)) containing the coefficients for the
        lower surface followed by the upper surface.
        """
        if not hasattr(self, "_cst"):
            self._compute_cst()
        return self._cst

    @property
    def cst_upper(self):
        """CST coefficients for the upper (suction) side of the airfoil

        Return:
            np.ndarray: Array of length (BP+1)
        """
        if not hasattr(self, "_cst"):
            self._compute_cst()
        return self._cst_upper

    @property
    def cst_lower(self):
        """CST coefficients for the lower (pressure) side of the airfoil

        Return:
            np.ndarray: Array of length (BP+1)
        """
        if not hasattr(self, "_cst"):
            self._compute_cst()
        return self._cst_lower

    def __call__(self, xinp, p_ar=None, te_upper=None, te_lower=None):
        """Compute coordinates for the airfoil

        Args:
            xinp (np.ndarray): Non-dimensional x-coordinate locations
            p_ar (np.ndarray): Non-dimensional perturbation of cst coefficients
            te_upper (double): Trailing edge thickness above camber line
            te_lower (double): Trailing edge thickness below camber line

        Return:
            tuple: (ylo, yup) Numpy arrays for the lower, upper y-coordinates
        """
        xco = np.asarray(xinp)
        telo = self.airfoil.te_lower if te_lower is None else te_lower
        teup = self.airfoil.te_upper if te_upper is None else te_upper
        amat = self.cst_matrix(xinp)
        cst_lower = self.cst_lower
        cst_upper = self.cst_upper
        if p_ar is not None:
            cst_lower = self.cst_lower * (1.0 + p_ar[:(1+self.order)])
            cst_upper = self.cst_upper * (1.0 + p_ar[(1+self.order):])
        ylo = np.dot(amat, cst_lower) + telo * xco
        yup = np.dot(amat, cst_upper) + teup * xco
        return (ylo, yup)

class AirfoilShape:
    """Representation of airfoil point data"""

    def __init__(self, xco, yco, shape_class='airfoil'):
        """
        Args:
            xco (np.ndarray): Array of x-coordinates
            yco (np.ndarray): Array of y-coordinates
            shape_class (string): Airfoil Shape type
        """
        xlo = np.min(xco)
        xhi = np.max(xco)

        #: Chord length based on input data
        self.chord = (xhi - xlo)
        #: Normalized x-coordinate array
        self.xco = (xco - xlo) / self.chord
        #: Normalized y-coordinate array
        self.yco = yco / self.chord

        # Leading edge index
        le_idx = np.argmin(self.xco)
        # Determine orientation of the airfoil shape
        y1avg = np.average(self.yco[:le_idx])
        # Flip such that the pressure side is always first
        if y1avg > 0.0:
            self.xco = self.xco[::-1]
            self.yco = self.yco[::-1]

        self._le = np.argmin(self.xco)

        self.shape_class = shape_class

    @classmethod
    def from_cst_parameters(cls, cst_lower, te_lower, cst_upper, te_upper,
                            n1=0.5, n2=1.0):
        """Create airfoil from CST parameters
        Args:
            cst_lower (np.ndarray): Array of lower surface CST parameters
            cst_upper (np.ndarray): Array of upper surface CST parameters
            te_lower (double): Lower surface trailing edge y coordinate
            te_upper (double): Upper surface trailing edge y coordinate
            n1 (double): N1 parameter for CST
            n2 (double): N2 parameter for CST
        """
        ccst = CSTAirfoil.from_cst_parameters(cls,cst_lower,cst_upper,n1,n2)
        x_c = -np.cos(np.arange(0,np.pi+0.005,np.pi*0.005))*0.5+0.5
        yl,yu = ccst(x_c, te_upper=te_upper, te_lower=te_lower)
        xco = np.append(x_c[::-1],x_c[1:])
        yco = np.append(yl[::-1],yu[1:])
        self = AirfoilShape(xco,yco)
        self._cst = ccst
        return self

    @classmethod
    def from_txt_file(cls, coords_file):
        """Load airfoil from a text file"""
        fpath = Path(coords_file).resolve()
        assert fpath.exists()
        xco, yco = np.loadtxt(fpath, unpack=True)
        self = AirfoilShape(xco, yco)
        return self


    @property
    def xupper(self):
        """Coordinates of the suction side"""
        return self.xco[self._le:]

    @property
    def yupper(self):
        """Coordinates of the suction side"""
        return self.yco[self._le:]

    @property
    def xlower(self):
        """Coordinates of the pressure side"""
        return self.xco[:self._le+1]

    @property
    def ylower(self):
        """Coordinates of the pressure side"""
        return self.yco[:self._le+1]

    @property
    def te_upper(self):
        """Trailing edge thickness on suction side"""
        return self.yco[-1]

    @property
    def te_lower(self):
        """Trailing edge thickness on pressure side"""
        return self.yco[0]

    def cst(self, order=8):
        """Return CST representation of the airfoil"""
        if not hasattr(self, "_cst"):
            self._cst = CSTAirfoil(self, order, self.shape_class)
        return self._cst

    def n1(self):
        """Return n1 parameter for airfoil"""
        if not hasattr(self, "_cst"):
            self._cst = CSTAirfoil(self, order, self.shape_class)
        return self._cst.n1

    def n2(self):
        """Return n2 parameter for airfoil"""
        if not hasattr(self, "_cst"):
            self._cst = CSTAirfoil(self, order, self.shape_class)
        return self._cst.n2


    def __call__(self, xinp):
        """Return interpolated y-coordinates for an airfoil

        Args:
            xinp (np.ndarray): Non-dimensional x-coordinate locations

        Return:
            tuple: (xco, ylo, yup) Dimensional (lower, upper) y-coordinates
        """
        afcst = self.cst()
        (ylo, yup) = afcst(xinp)
        return (xinp * self.chord, ylo * self.chord, yup * self.chord)

    def perturb(self, xinp, p_ar):
        """ Return perturbed y-coordinates for an airfoil by perturbing
        the cst coefficients

        Args:
            xinp (np.ndarray): Non-dimensional x-coodinate locations
            p_ar (np.ndarray): Non-dimensional perturbation

        Return:
            tuple: (xco, ylo, yup) Dimensional (lower, upper) y-coordinates
        """

        afcst = self.cst()
        (ylo, yup)= afcst(xinp, p_ar)
        return (xinp * self.chord, ylo * self.chord, yup * self.chord)

