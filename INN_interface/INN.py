import os
import psdr
import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.optimize import lsq_linear
from INN_interface.inv_net import InvNet
from INN_interface.utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

class INN():
    def __init__(self, afs=['DU21_A17', 'DU25_A17', 'DU30_A17', 'DU35_A17', 'FFA_W3_211', 'FFA_W3_241',
                            'FFA_W3_270blend', 'FFA_W3_301', 'FFA_W3_330blend', 'FFA_W3_360',
                            'FFA_W3_360GF', 'NACA64_A17']):
        '''
            DESCRIPTION 

            Parameters:
                afs           - list of INN models to inlude for analysis
                                each model is defined the baseline airfoil around which perturbations occur
                Ws            - dictionary of numpy arrays defining input dimension reduction
                scale_factors - dictionary of scale_factor dictionaries containing normalization factors for each model
                models        - dictionary of invertible neural network models

            Methods:
                generate_polars() - Given input CST parameters (N_shapes x 20) and (1 or N) Reynolds numbers, produce the 
                                    polar for cd and cl over a range of angles of attack (this can be specified as well)
                inverse_design()  - Given conditional quantities, generate N shapes that approximately produce those quantities
                                    The shapes can be processed by a forward evaluation of the associated INN to rank the shapes
        '''
        super(INN, self).__init__()
        # Fixed input/output dimensions [Could be changed to be flexible later]
        xM, yM, cM, fM, zM, lM = 5, 1, 1, 3, 1, 1
        x_in = tf.keras.Input(shape=(xM,))
        y_in = tf.keras.Input(shape=(yM,))
        c_in = tf.keras.Input(shape=(cM,))
        f_in = tf.keras.Input(shape=(fM,))
        z_in = tf.keras.Input(shape=(zM,))
        l_in = tf.keras.Input(shape=(lM,))

        self.Ws = {}
        self.scale_factors = {}
        self.models = {}
        self.KL_basis = {}
        for af in afs:
            this_directory = os.path.abspath(os.path.dirname(__file__))
            airfoil_path = os.path.join(this_directory, 'model/'+af)
            self.Ws[af] = np.load(airfoil_path + '/U.npy')
            self.scale_factors[af] = load_scale_factors(af)
            self.models[af] = InvNet(x_in, y_in, c_in, f_in, z_in, l_in,
                                     input_shape=tf.TensorShape([xM+yM]),
                                     n_layers=15, W=self.Ws[af],
                                     scale_factors=self.scale_factors[af],
                                     model_path=airfoil_path + '/inn.h5')
            self.KL_basis[af] = airfoil_path+'/polar_KL.h5'

    def generate_polars(self, cst, Re, alpha=np.arange(-4, 20.1, 0.25), af=None, KL_basis=True):
        cst = cst.reshape((1, -1)) if cst.ndim == 1 else cst
        af = self.identify_airfoil(cst=cst) if af is None else af
        if isinstance(af, str):
            af = [af for _ in range(cst.shape[0])]

        N_shapes, N_angles = cst.shape[0], alpha.size

        # Convert Re number to numpy array
        if np.isscalar(Re):
            Re = np.repeat(np.array([[Re]]), cst.shape[0], axis=0)
        Re = Re.reshape((N_shapes, 1))

        cd, cl = np.zeros((N_shapes, N_angles)), np.zeros((N_shapes, N_angles))
        for i in range(N_shapes):
            cst_i = cst[i, :].reshape((1, -1))
            Re_i = Re[i].reshape((1, 1))

            # Remove trailing edge factors if held constant
            if 'trailing_edge' in self.scale_factors[af[i]]:
                cst_i = cst_i[:, :-2]

            # Matches input CST parameters to sweep of angles of attack
            cst_i = np.repeat(cst_i, N_angles, axis=0)
            alpha = alpha.reshape((-1, 1))

            # Perform input normalizations and dimension reductions before passing to network
            x = np.concatenate((cst_i, alpha), axis=1)
            x, _ = norm_data(x, self.models[af[i]].scale_factors['x'], scale_type='minmax')
            x = np.concatenate((x[:, :-1]@self.Ws[af[i]], x[:, -1:]), axis=1)
            if self.models[af[i]].xM > x.shape[1]:
                x = np.concatenate((x, np.zeros((x.shape[0], self.models[af[i]].xM-x.shape[1]))), axis=1)

            # Normalize Renolds number
            y, _ = norm_data(np.repeat(Re_i, N_angles, axis=0), self.models[af[i]].scale_factors['y'])

            # Evalutate forward model and return values to physical space
            _, _, f_out, _ = self.models[af[i]].eval_forward(x, y)
            f_out = unnorm_data(f_out, scale_factor=self.scale_factors[af[i]]['f'])
            f_out = f_out.numpy()
            f_out[:, 0] = np.power(10., f_out[:, 0])

            f_out = f_out.reshape((1, N_angles, self.models[af[i]].fM))
            
            # Obtain coefficients of lift and drag
            cd_i, clcd_i = f_out[:, :, 0], f_out[:, :, 1]
            cl_i = cd_i*clcd_i

            if KL_basis:
                Re_KL, alpha_KL = 1e6*np.arange(3., 12.1, 3.), np.arange(-4., 20.1, 1.)
                Re_KL = Re_KL[np.argmin(abs(Re_KL - Re[i]))]
                with h5py.File(self.KL_basis[af[i]], 'r') as f:
                    alpha_KL = f['alpha'][()]
                    cd_KL = f['cd/re{}/KL_basis'.format(int(Re_KL))][()]
                    cd_KL_bounds = f['cd/re{}/KL_bounds'.format(int(Re_KL))][()]
                    cl_KL = f['cl/re{}/KL_basis'.format(int(Re_KL))][()]
                    cl_KL_bounds = f['cl/re{}/KL_bounds'.format(int(Re_KL))][()]
                
                cd_interp = np.zeros((N_angles, cd_KL.shape[1]))
                cl_interp = np.zeros((N_angles, cl_KL.shape[1]))
                for j in range(cd_KL.shape[1]):
                    f_interp = interpolate.interp1d(alpha_KL, cd_KL[:, j])
                    cd_interp[:, j] = f_interp(alpha)[:, 0]

                    f_interp = interpolate.interp1d(alpha_KL, cl_KL[:, j])
                    cl_interp[:, j] = f_interp(alpha)[:, 0]

                coef_cd = lsq_linear(cd_interp, cd_i[0, :], 
                                     bounds=(cd_KL_bounds[:, 0], cd_KL_bounds[:, 1]))
                cd_i = cd_interp @ coef_cd.x

                coef_cl = lsq_linear(cl_interp, cl_i[0, :],
                                     bounds=(cl_KL_bounds[:, 0], cl_KL_bounds[:, 1]))
                cl_i = cl_interp @ coef_cl.x

            cd[i, :] = cd_i
            cl[i, :] = cl_i

        return cd, cl

    def inverse_design(self, cd, clcd, stall_margin, thickness, Re, 
                       af='DU25_A17', z=None, N=1, process_samples=True):

        # Determine dimension of zero paddings of input space (artifact of INN architecutre)
        xM_padding = (self.models[af].cM+self.models[af].fM+self.models[af].zM) - (self.Ws[af].shape[1]+1)

        # Normalize and format conditional inputs for inverse model evaluation
        y_val, c_val = np.array([Re]), np.array([thickness])
        f_val = np.array([np.log10(cd), clcd, stall_margin] + [0. for _ in range(self.models[af].fM-3)])

        y_val, _ = norm_data(y_val.reshape((-1, self.models[af].yM)), self.models[af].scale_factors['y'])
        c_val, _ = norm_data(c_val.reshape((-1, self.models[af].cM)), self.models[af].scale_factors['c'])
        f_val, _ = norm_data(f_val.reshape((-1, self.models[af].fM)), self.models[af].scale_factors['f'])

        if (z is None) or (np.isscalar(z)):
            cheby_mean = False

            # If process_samples, then we must oversample the inverse direction to find best shapes
            NN = 10*N if process_samples else N

            # Run inverse model with randomly sampled latent variables (z)
            y_val = tf.repeat(y_val, NN, axis=0)
            c_val = tf.repeat(c_val, NN, axis=0)
            f_val = tf.repeat(f_val, NN, axis=0)

            if z is None:
                z_val = tf.random.normal([NN, self.models[af].zM], dtype=tf.float64)
            if np.isscalar(z):
                tf.random.set_seed(z)
                z_val = tf.random.normal([1, self.models[af].zM], dtype=tf.float64)
                z_val = tf.repeat(z_val, NN, axis=0)
                cheby_mean = True
        else:
            # If z is provided, then override N and process_samples and use given z
            cheby_mean = True
            if not tf.is_tensor(z):
                z = tf.convert_to_tensor(z)
            if tf.rank(z) == 1:
                assert z.shape[0] == self.models[af].zM
                z = tf.reshape(z, [1, self.models[af].zM])
            NN = z.shape[0]

            y_val = tf.repeat(y_val, NN, axis=0)
            c_val = tf.repeat(c_val, NN, axis=0)
            f_val = tf.repeat(f_val, NN, axis=0)
            z_val = z

            process_samples = False

        x_inv, _ = self.models[af].eval_inverse(y_val, c_val, f_val, z_val)
        
        # Sort generated shapes by errors in network forward prediction
        if process_samples:
            x_inv = self.sort_by_errs(x_inv, y_val, c_val, f_val, af)[:N, :]

        # Map reduced dimension and normalized inputs back to physical space
        x_inv = x_inv[:, :-xM_padding]
        cst, alpha = self.recover_full_cst(x_inv, af, cheby_mean=cheby_mean)

        return cst, alpha

    def hierarchical_design(self):
        # Function that will handle sampling from the various baseline models
        pass

    def identify_airfoil(self, cst=None):
        this_directory = os.path.abspath(os.path.dirname(__file__))
        af_cst = np.load(os.path.join(this_directory, 'model/baseline_cst.npz'))

        D_min = [1e6 for _ in range(cst.shape[0])]
        af_opt = [None for _ in range(cst.shape[0])]
        for af in af_cst:
            cst_diff = abs(cst - af_cst[af])
            cst_diff -= 0.2*abs(af_cst[af])
            tf = (cst_diff < 0.)
            cst_diff = abs(cst_diff)
            cst_diff[tf] = 0.

            D = np.sum(cst_diff**2, axis=1)

            tf = [(af_i is None) for af_i in af_opt] | (D < D_min)

            af_opt = [af if tf_i else af_opt[i] for i, tf_i in enumerate(tf)]
            D_min = [D[i] if tf_i else D_min[i] for i, tf_i in enumerate(tf)]

        return af_opt

    def recover_full_cst(self, x, af, cheby_mean=False):
        N = x.shape[0]
        m = self.Ws[af].shape[0]
        x_shape, alpha = x[:, :-1].numpy(), x[:, -1:].numpy()

        # Uses a hit-and-run sampler to map reduced dimensional inputs back into full space
        cst = np.zeros((N, m))
        domain = psdr.BoxDomain(-1.01*np.ones(m), 1.01*np.ones(m))
        for i in range(N):
            con_domain = domain.add_constraints(A_eq=self.Ws[af].T, b_eq=x_shape[i, :])
            if cheby_mean:
                np.random.seed(42)

            keep_trying, failed_attempts = True, 0
            while keep_trying:
                try:
                    cst_i = con_domain.sample(10)
                    keep_trying = False
                except:
                    failed_attempts += 1
                    if failed_attempts > 10:
                        assert False, 'Too many failed attempts'
            if cheby_mean:
                cst[i, :] = cst_i[0, :]
            else:
                cst[i, :] = cst_i[-1, :]

        # Maps normalilzed CST parameters back to physical space
        x = np.concatenate((cst, alpha), axis=1)
        x = unnorm_data(x, scale_factor=self.scale_factors[af]['x'], scale_type='minmax')

        cst, alpha = x[:, :-1], x[:, -1:]

        # Appends the trailing edge thickness that was removed if non-informative in dataset
        if 'trailing_edge' in self.scale_factors[af]:
            trailing_edge = np.array(self.scale_factors[af]['trailing_edge'][:2])
            trailing_edge = tf.repeat(trailing_edge.reshape((1, -2)), N, axis=0)
            cst = np.concatenate((cst, trailing_edge), axis=1)
       
        return cst, alpha

    def approx_errors(self, x, y, c, f, af):
        # Run forward model on given shapes
        y_out, c_out, f_out, _ = self.models[af].eval_forward(x, y)

        y = unnorm_data(y, scale_factor=self.scale_factors[af]['y'])
        c = unnorm_data(c, scale_factor=self.scale_factors[af]['c'])
        f = unnorm_data(f, scale_factor=self.scale_factors[af]['f'])
        y, c, f = y.numpy(), c.numpy(), f.numpy()
        
        y_out = unnorm_data(y_out, scale_factor=self.scale_factors[af]['y'])
        c_out = unnorm_data(c_out, scale_factor=self.scale_factors[af]['c'])
        f_out = unnorm_data(f_out, scale_factor=self.scale_factors[af]['f'])
        y_out, c_out, f_out = y_out.numpy(), c_out.numpy(), f_out.numpy()

        f[:, 0], f_out[:, 0] = np.power(10., f[:, 0]), np.power(10., f_out[:, 0])

        # Compute relative errors for each output
        y_err = np.sum(np.abs((y - y_out)/y), axis=1)
        c_err = np.sum(np.abs((c - c_out)/c), axis=1)
        f_err = np.sum(np.abs((f - f_out)/f), axis=1)

        return y_err, c_err, f_err
        
    def sort_by_errs(self, x, y, c, f, af):
        # Sort shapes by total errors
        y_err, c_err, f_err = self.approx_errors(x, y, c, f, af)

        idx = np.argsort(y_err + c_err + f_err)

        x = tf.convert_to_tensor(x.numpy()[idx, :])

        return x


