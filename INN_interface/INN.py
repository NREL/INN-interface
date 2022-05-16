import os
import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from INN_interface.inv_net import InvNet
from INN_interface.utils import *
from INN_interface.Grassmann import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

class INN():
    def __init__(self):
        '''
            Parameters:
                model         - invertible neural network models based on PGA shape representations
                scale_factors - dictionaries containing normalization factors for each quantity
                Vh            - PGA coordinate directions in tanget space of karcher mean
                karcher_mean  - centered shape about which PGA perturbations are defined
                KL_basis      - basis functions and limits for KL expansion of aerodynamic polars
                

            Methods:
                generate_polars() - Given input shape data (XY landmarks, PGA coefficients, or CST coefficients) 
                                    and (1 or N) Reynolds numbers, produce the polar curves for cd and cl over a 
                                    range of angles of attack (which can be specified as well)
                inverse_design()  - Given conditional quantities, generate N shapes that approximately produce 
                                    those quantities
        '''
        super(INN, self).__init__()
        # Fixed input/output dimensions [Could be changed to be flexible later]
        xM, yM, cM, fM, zM, lM = 7, 1, 1, 3, 3, 1
        x_in = tf.keras.Input(shape=(xM,))
        y_in = tf.keras.Input(shape=(yM,))
        c_in = tf.keras.Input(shape=(cM,))
        f_in = tf.keras.Input(shape=(fM,))
        z_in = tf.keras.Input(shape=(zM,))
        l_in = tf.keras.Input(shape=(lM,))

        this_directory = os.path.abspath(os.path.dirname(__file__))
        airfoil_path = os.path.join(this_directory, 'model/PGA')

        self.scale_factors = load_scale_factors('PGA')
        self.model = InvNet(x_in, y_in, c_in, f_in, z_in, l_in,
                            input_shape=tf.TensorShape([xM+yM]),
                            n_layers=15, W=np.eye(xM-1),
                            scale_factors=self.scale_factors,
                            model_path=airfoil_path + '/inn.h5')

        self.Vh = np.load(airfoil_path+'/Vh.npy')
        self.karcher_mean = np.load(airfoil_path+'/karcher_mean.npy')
        self.KL_basis = airfoil_path+'/polar_KL.h5'

    def pga_to_cst(self, pga_in):
        # Convert PGA inputs (N_shapes x 6) to CST inputs (N_shapes x 20)
        pga = np.copy(pga_in)
        cst = self.shape_to_cst(self.pga_to_shape(pga))

        return cst

    def cst_to_pga(self, cst_in):
        # Convert CST inputs (N_shapes x 20) to PGA inputs (N_shapes x 6)
        cst = np.copy(cst_in)
        pga = self.shape_to_pga(self.cst_to_shape(cst))

        return pga

    def shape_to_pga(self, landmarks_in):
        # Convert XY landmarks (N_shapes x 401 x 2) to PGA inputs (N_shapes x 6)
        landmarks = np.copy(landmarks_in)
        landmarks = np.expand_dims(landmarks, axis=0) if landmarks.ndim == 2 else landmarks
        
        y_te_lower, y_te_upper = np.copy(landmarks[:, :1, 1]), np.copy(landmarks[:, -1:, 1])

        landmarks[:, :200, 1] -= landmarks[:, :200, 0]*y_te_lower
        landmarks[:, 200:, 1] -= landmarks[:, 200:, 0]*y_te_upper

        landmarks_gr, M_gr, b_gr = landmark_affine_transform(landmarks)
        gr_coords = get_PGA_coordinates(landmarks_gr, self.karcher_mean, self.Vh)
        
        pga = np.concatenate((gr_coords, M_gr[:, 1, 1].reshape((-1, 1)), y_te_upper-y_te_lower), axis=1)

        return pga
    
    def pga_to_shape(self, pga_in):
        # Convert PGA inputs (N_shapes x 6) to XY landmarks (N_shapes x 401 x 2)
        pga = np.copy(pga_in)
        pga = pga.reshape((1, -1)) if pga.ndim == 1 else pga

        x_c = -np.cos(np.arange(0, np.pi+0.005, np.pi*0.005))*0.5+0.5

        landmarks = np.empty((pga.shape[0], 401, 2))
        for i, pga_i in enumerate(pga):
            M = np.array([[-7.08867942e+00, -7.99187803e-04],
                          [-0.08200894864926603, pga_i[-2]]])
            b = np.array([[0.50124688, 0.]])
            gr_shape_i = perturb_gr_shape(self.Vh, self.karcher_mean, 
                                          pga_i[:-2].reshape((1, -1)), scale=1)
            landmark_i = gr_shape_i @ M.T + b
            
            x_min = np.min(landmark_i[:, 0])
            x_te_l = landmark_i[0, 0]
            x_te_u = landmark_i[-1, 0]
            landmark_i[:, 0] = (landmark_i[:, 0] - x_min)
            landmark_i[:201, :] = landmark_i[:201, :]/(x_te_l - x_min)
            landmark_i[200:, :] = landmark_i[200:, :]/(x_te_u - x_min)
            y_le, y_te_lower = landmark_i[200, 1], landmark_i[0, 1]
            landmark_i[:, 1] = (landmark_i[:, 1] - y_le)
            landmark_i[:, 1] = (landmark_i[:, 1] + landmark_i[:, 0]*(y_le-y_te_lower))

            ff = interp1d(landmark_i[:201, 0], landmark_i[:201, 1], kind='cubic')
            yl_c = ff(np.flip(x_c))
            ff = interp1d(landmark_i[200:, 0], landmark_i[200:, 1], kind='cubic')
            yu_c = ff(x_c)

            landmarks[i, :, 0] = np.concatenate((np.flip(x_c[1:]), x_c), axis=0)
            landmarks[i, :, 1] = np.concatenate((yl_c[:-1], yu_c), axis=0)
        
            landmarks[i, :200, 1] -= landmarks[i, :200, 0]*(0.5*pga_i[-1])
            landmarks[i, 200:, 1] += landmarks[i, 200:, 0]*(0.5*pga_i[-1])
            
        return landmarks

    def shape_to_cst(self, landmarks_in):
        # Convert XY landmarks (N_shapes x 401 x 2) to CST inputs (N_shapes x 20)
        landmarks = np.copy(landmarks_in)
        landmarks = np.expand_dims(landmarks, axis=0) if landmarks.ndim == 2 else landmarks

        y_te_lower, y_te_upper = landmarks[:, :1, 1], landmarks[:, -1:, 1]

        cst = np.empty((landmarks.shape[0], 20))
        for i, xy in enumerate(landmarks):
            A = self.cst_matrix(xy[:200, 0], 0.5, 1.0, 8)
            A = np.hstack((A, xy[:200, 0].reshape(-1, 1)))
            out = lsq_linear(A, xy[:200, 1])
            cst_lower = out.x[:9]

            A = self.cst_matrix(xy[200:, 0], 0.5, 1.0, 8)
            A = np.hstack((A, xy[200:, 0].reshape(-1, 1)))
            out = lsq_linear(A, xy[200:, 1])
            cst_upper = out.x[:9]

            cst[i, :] = np.concatenate((cst_lower, cst_upper, 
                                        y_te_lower[i, :], y_te_upper[i, :]), axis=0)

        return cst

    def cst_to_shape(self, cst_in):
        # Convert CST inputs (N_shapes x 20) to XY landmarks (N_shapes x 401 x 2)
        cst = np.copy(cst_in)
        cst = cst.reshape((1, -1)) if cst.ndim == 1 else cst

        n_half = int(401 / 2)
        x_c = -np.cos(np.linspace(0, np.pi, n_half + 1)) * 0.5 + 0.5
        landmarks = np.empty((cst.shape[0], 401, 2))
        for i, cst in enumerate(cst):
            cst_lower = np.append(cst[0:9], cst[-2])
            cst_upper = np.append(cst[9:18], cst[-1])

            order = np.size(cst_lower) - 2
            amat = self.cst_matrix(x_c, 0.5, 1.0, order)
            amat = np.hstack((amat, x_c.reshape(-1, 1)))

            y_lower = np.dot(amat, cst_lower)
            y_upper = np.dot(amat, cst_upper)

            x = np.hstack((x_c[::-1], x_c[1:])).reshape(-1, 1)
            y = np.hstack((y_lower[::-1], y_upper[1:])).reshape(-1, 1)

            landmarks[i] = np.hstack((x, y))

        return landmarks

    def cst_matrix(self, x, n1=0.5, n2=1.0, order=8):
        # Create CST matrix for fitting CST parameters to airfoil shape
        x = np.asarray(x)
        class_function = np.power(x, n1) * np.power((1.0 - x), n2)

        K = comb(order, range(order + 1))
        shape_function = np.empty((order + 1, x.shape[0]))
        for i in range(order + 1):
            shape_function[i, :] = K[i] * np.power(x, i) * np.power((1.0 - x), (order - i))

        return (class_function * shape_function).T

    def sort_by_errs(self, x_inv, y_inv, y, c, f, z, N=1):
        # Sort shapes by total errors
        y_out, c_out, f_out, _ = self.model.eval_forward(x_inv, y)

        y = unnorm_data(y, scale_factor=self.scale_factors['y'])
        c = unnorm_data(c, scale_factor=self.scale_factors['c'])
        f = unnorm_data(f, scale_factor=self.scale_factors['f'])
        y, c, f = y.numpy(), c.numpy(), f.numpy()
        
        y_inn = unnorm_data(y_inv, scale_factor=self.scale_factors['y'])
        y_out = unnorm_data(y_out, scale_factor=self.scale_factors['y'])
        c_out = unnorm_data(c_out, scale_factor=self.scale_factors['c'])
        f_out = unnorm_data(f_out, scale_factor=self.scale_factors['f'])
        y_out, c_out, f_out = y_out.numpy(), c_out.numpy(), f_out.numpy()
        f[:, 0], f_out[:, 0] = np.power(10., f[:, 0]), np.power(10., f_out[:, 0])

        x_inv, y_inv, z = x_inv.numpy(), y_inv.numpy(), z.numpy()

        # Compute relative errors for each output
        y1_err = np.sum(np.abs((y - y_inn)/y), axis=1)
        y2_err = np.sum(np.abs((y_out - y_inn)/y), axis=1)
        y3_err = np.sum(np.abs((y - y_out)/y), axis=1)
        idx = np.argsort(y1_err + y2_err + y3_err)[:np.minimum(x_inv.shape[0], 3*N)]
        
        x_inv = x_inv[idx, :]
        y_inv = y_inv[idx, :]
        z = z[idx, :]

        c_err = np.sum(np.abs((c - c_out)/c), axis=1)[idx]
        f_err = np.sum(np.abs((f - f_out)/f), axis=1)[idx]
        idx = np.argsort(c_err + f_err)

        x_inv = x_inv[idx[:N], :]
        y_inv = y_inv[idx[:N], :]
        z = z[idx[:N], :]

        return x_inv, y_inv, z

    def generate_polars(self, X, Re, alpha=np.arange(-4, 20.1, 0.25),
                              data_format='XY', KL_basis=True):
        '''
            Parameters:
                X            - shape data (in appropriate form according to data_format)
                Re           - Reynolds number
                alpha        - range of angles of attack to compute polars
                data_format  - string specifying format for shape representations
                                  'XY': (N_shapes x 401 x 2) landmarks around the airfoil
                                        starting at the trailing edge and going clockwise
                                 'PGA': (N_shapes x 6) coefs for the PGA shape representation
                                 'CST': (N_shapes x 20) coefs for the PGA shape representation
                KL_basis     - True/False whether to process polar data through KL basis expansion
        '''

        if data_format.upper() == 'XY':
            X = X.reshape((1, -1)) if X.ndim == 2 else X
            X = self.shape_to_pga(X)
        elif data_format.upper() == 'CST':
            X = X.reshape((1, -1)) if X.ndim == 1 else X
            X = self.cst_to_pga(X)
        else:
            error_message = 'data_format must be XY, CST, or PGA'
            assert data_format.upper() == 'PGA', error_message

            X = X.reshape((1, -1)) if X.ndim == 1 else X

        N_shapes, N_angles = X.shape[0], alpha.size

        # Convert Re number to numpy array
        if np.isscalar(Re):
            Re = np.repeat(np.array([[Re]]), X.shape[0], axis=0)
        Re = Re.reshape((N_shapes, 1))

        cd, cl = np.empty((N_shapes, N_angles)), np.empty((N_shapes, N_angles))
        for i in range(N_shapes):
            X_i = X[i, :].reshape((1, -1))
            Re_i = Re[i].reshape((1, 1))

            # Matches input PGA parameters to sweep of angles of attack
            X_i = np.repeat(X_i, N_angles, axis=0)
            alpha = alpha.reshape((-1, 1))

            # Perform input normalizations and dimension reductions before passing to network
            x = np.concatenate((X_i, alpha), axis=1)
            x, _ = norm_data(x, self.scale_factors['x'])
            
            # Normalize Renolds number
            y, _ = norm_data(np.repeat(Re_i, N_angles, axis=0), self.scale_factors['y'])

            # Evalutate forward model and return values to physical space
            _, _, f_out, _ = self.model.eval_forward(x, y)

            f_out = unnorm_data(f_out, scale_factor=self.scale_factors['f'])
            f_out = f_out.numpy()
            f_out[f_out[:, 0] > 5., 0] = 5.
            f_out[:, 0] = np.power(10., f_out[:, 0])

            f_out = f_out.reshape((1, N_angles, self.model.fM))
            
            # Obtain coefficients of lift and drag
            cd_i, clcd_i = f_out[:, :, 0], f_out[:, :, 1]
            cl_i = cd_i*clcd_i

            if KL_basis:
                Re_KL, alpha_KL = 1e6*np.arange(3., 12.1, 3.), np.arange(-4., 20.1, 1.)
                Re_KL = Re_KL[np.argmin(abs(Re_KL - Re[i]))]
                with h5py.File(self.KL_basis, 'r') as f:
                    alpha_KL = f['alpha'][()]
                    cd_KL = f['cd/re{}/KL_basis'.format(int(Re_KL))][()]
                    cd_KL_bounds = f['cd/re{}/KL_bounds'.format(int(Re_KL))][()]
                    cl_KL = f['cl/re{}/KL_basis'.format(int(Re_KL))][()]
                    cl_KL_bounds = f['cl/re{}/KL_bounds'.format(int(Re_KL))][()]
                
                cd_interp = np.empty((N_angles, cd_KL.shape[1]))
                cl_interp = np.empty((N_angles, cl_KL.shape[1]))
                for j in range(cd_KL.shape[1]):
                    f_interp = interpolate.interp1d(alpha_KL, cd_KL[:, j])
                    cd_interp[:, j] = f_interp(alpha)[:, 0]

                    f_interp = interpolate.interp1d(alpha_KL, cl_KL[:, j])
                    cl_interp[:, j] = f_interp(alpha)[:, 0]

                n_KL_terms = 4
                cd_interp, cd_KL_bounds = cd_interp[:, :n_KL_terms], cd_KL_bounds[:n_KL_terms, :]
                cl_interp, cl_KL_bounds = cl_interp[:, :n_KL_terms], cl_KL_bounds[:n_KL_terms, :]
                coef_cd = lsq_linear(cd_interp, np.log10(cd_i[0, :]), 
                                     bounds=(cd_KL_bounds[:, 0], cd_KL_bounds[:, 1]))
                cd_i = np.power(10., cd_interp @ coef_cd.x)

                coef_cl = lsq_linear(cl_interp, cl_i[0, :],
                                     bounds=(cl_KL_bounds[:, 0], cl_KL_bounds[:, 1]))
                cl_i = cl_interp @ coef_cl.x

            cd[i, :] = cd_i
            cl[i, :] = cl_i

        return cd, cl

    def inverse_design(self, cd, clcd, stall_margin, thickness, Re, 
                       z=None, N=1, process_samples=True, return_z=True, data_format='XY'):
        '''
            Parameters:
                cd               - coefficient drag
                clcd             - coefficient lift / coefficient drag
                stall_margin     - stall margin
                thickness        - maximum thickness to chord ratio
                Re               - Reynolds number
                z                - flag for how to set latent variables
                                       None: randomly sample z
                               scalar value: fix seed at specific value before drawing z
                     NumPy/TensorFlow array: quantities for z
                N                - number of designs to generate
                process_samples  - True/False whether to filter out for best designs
                return_z         - True/False whether to return corresponding latent variables
                data_format      - string specifying format for returned shape representations
                                     'XY': (N_shapes x 401 x 2) landmarks around the airfoil
                                           starting at the trailing edge and going clockwise
                                    'PGA': (N_shapes x 6) coefs for the PGA shape representation
                                    'CST': (N_shapes x 20) coefs for the PGA shape representation
        '''

        # Normalize and format conditional inputs for inverse model evaluation
        y_val, c_val = np.array([Re]), np.array([thickness])
        f_val = np.array([np.log10(cd), clcd, stall_margin] + [0. for _ in range(self.model.fM-3)])

        y_val, _ = norm_data(y_val.reshape((-1, self.model.yM)), self.scale_factors['y'])
        c_val, _ = norm_data(c_val.reshape((-1, self.model.cM)), self.scale_factors['c'])
        f_val, _ = norm_data(f_val.reshape((-1, self.model.fM)), self.scale_factors['f'])

        if (z is None):
            # If process_samples, then we must oversample the inverse direction to find best shapes
            NN = np.maximum(10*N, 100) if process_samples else N

            # Run inverse model with randomly sampled latent variables (z)
            y_val = tf.repeat(y_val, NN, axis=0)
            c_val = tf.repeat(c_val, NN, axis=0)
            f_val = tf.repeat(f_val, NN, axis=0)

            z_val = tf.random.normal([NN, self.model.zM], dtype=tf.float64)
                
        elif np.isscalar(z):
            y_val = tf.repeat(y_val, N, axis=0)
            c_val = tf.repeat(c_val, N, axis=0)
            f_val = tf.repeat(f_val, N, axis=0)

            tf.random.set_seed(z)
            z_val = tf.random.normal([N, self.model.zM], dtype=tf.float64)

            process_samples = False

        else:
            # If z is provided, then override N and process_samples and use given z
            if not tf.is_tensor(z):
                z = tf.convert_to_tensor(z)
            if tf.rank(z) == 1:
                assert z.shape[0] == self.model.zM
                z = tf.reshape(z, [1, self.model.zM])
            if z.shape[0] == 1:
                z = tf.repeat(z, N, axis=0)
            NN = z.shape[0]

            y_val = tf.repeat(y_val, NN, axis=0)
            c_val = tf.repeat(c_val, NN, axis=0)
            f_val = tf.repeat(f_val, NN, axis=0)
            z_val = z

            process_samples = False

        x_inv, y_inv = self.model.eval_inverse(y_val, c_val, f_val, z_val)

        # Sort generated shapes by errors in network forward prediction
        if process_samples:
            x_inv, y_inv, z_val = self.sort_by_errs(x_inv, y_inv,
                                                    y_val, c_val, f_val, z_val,
                                                    N=N)
        else:
            x_inv, y_inv, z_val = x_inv.numpy(), y_inv.numpy(), z_val.numpy()

        # Map reduced dimension and normalized inputs back to physical space
        x_inv = unnorm_data(x_inv, scale_factor=self.scale_factors['x'])
        y_inv = unnorm_data(y_inv, scale_factor=self.scale_factors['y'])
        
        X, alpha = x_inv[:, :-1], x_inv[:, -1:]
            
        if data_format.upper() == 'XY':
            X = self.pga_to_shape(X)
        elif data_format.upper() == 'CST':
            X = self.pga_to_cst(X)
        else:
            error_message = 'data_format must be XY, CST, or PGA'
            assert data_format.upper() == 'PGA', error_message

        if return_z:
            return X, alpha, y_inv, z_val
        else:
            return X, alpha, y_inv


