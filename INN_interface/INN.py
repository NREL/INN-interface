import os
import psdr
import numpy as np
import tensorflow as tf
from inv_net import InvNet
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

class INN():
    def __init__(self, afs=['DU25_A17']):
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
        for af in afs:
            self.Ws[af] = np.load('model/'+af+'/U.npy')
            self.scale_factors[af] = load_scale_factors(af)
            self.models[af] = InvNet(x_in, y_in, c_in, f_in, z_in, l_in,
                                     input_shape=tf.TensorShape([xM+yM]),
                                     n_layers=15, W=self.Ws[af],
                                     scale_factors=self.scale_factors[af],
                                     model_path='model/'+af+'/inn.h5')

    def generate_polars(self, cst, Re, alpha=np.arange(-4, 20.1, 0.25)):
        # Baseline airfoil fixed as DU25... eventually this can be handled adaptively
        af = 'DU25_A17'

        N_shapes, N_angles = cst.shape[0], alpha.size

        # Convert Re number to numpy array
        if np.isscalar(Re):
            Re = np.repeat(np.array([[Re]]), cst.shape[0], axis=0)
        assert cst.shape[0] == Re.shape[0]

        # Remove trailing edge factors if held constant
        if 'trailing_edge' in self.scale_factors[af]:
            cst = cst[:, :-2]

        # Matches input CST parameters to sweep of angles of attack
        cst = np.repeat(cst, N_angles, axis=0)
        alpha = np.tile(alpha.reshape((-1, 1)), (N_shapes, 1))

        # Perform input normalizations and dimension reductions before passing to network
        x = np.concatenate((cst, alpha), axis=1)
        x, _ = norm_data(x, self.models[af].scale_factors['x'], scale_type='minmax')
        x = np.concatenate((x[:, :-1]@self.Ws[af], x[:, -1:]), axis=1)
        if self.models[af].xM > x.shape[1]:
            x = np.concatenate((x, np.zeros((x.shape[0], self.models[af].xM-x.shape[1]))), axis=1)

        # Normalize Renolds number
        y, _ = norm_data(np.repeat(Re, N_angles, axis=0), self.models[af].scale_factors['y'])

        # Evalutate forward model and return values to physical space
        _, _, f_out, _ = self.models[af].eval_forward(x, y)
        f_out = unnorm_data(f_out, scale_factor=self.scale_factors[af]['f'])
        f_out = f_out.numpy()
        f_out[:, 0] = np.power(10., f_out[:, 0])

        f_out = f_out.reshape((N_shapes, N_angles, self.models[af].fM))

        # Obtain coefficients of lift and drag
        cd, clcd = f_out[:, :, 0], f_out[:, :, 1]
        cl = cd*clcd

        return cd, cl

    def inverse_design(self, cd, clcd, stall_margin, thickness, Re, N=1, process_samples=True):
        # Baseline airfoil fixed as DU25... eventually this can be handled adaptively
        af = 'DU25_A17'

        # Determine dimension of zero paddings of input space (artifact of INN architecutre)
        xM_padding = (self.models[af].cM+self.models[af].fM+self.models[af].zM) - (self.Ws[af].shape[1]+1)

        # Normalize and format conditional inputs for inverse model evaluation
        y_val, c_val = np.array([Re]), np.array([thickness])
        f_val = np.array([np.log10(cd), clcd, stall_margin] + [0. for _ in range(self.models[af].fM-3)])

        y_val, _ = norm_data(y_val.reshape((-1, self.models[af].yM)), self.models[af].scale_factors['y'])
        c_val, _ = norm_data(c_val.reshape((-1, self.models[af].cM)), self.models[af].scale_factors['c'])
        f_val, _ = norm_data(f_val.reshape((-1, self.models[af].fM)), self.models[af].scale_factors['f'])

        # If process_sampes, then we must oversample the inverse direction to find best shapes
        NN = 10*N if process_samples else N

        # Run inverse model with randomly sampled latent variables (z)
        y_val = tf.repeat(y_val, NN, axis=0)
        c_val = tf.repeat(c_val, NN, axis=0)
        f_val = tf.repeat(f_val, NN, axis=0)
        z_val = tf.random.normal([NN, self.models[af].zM], dtype=tf.float64)
        x_inv, _ = self.models[af].eval_inverse(y_val, c_val, f_val, z_val)
        
        # Sort generated shapes by errors in network forward prediction
        if process_samples:
            x_inv = self.sort_by_errs(x_inv, y_val, c_val, f_val, af)[:N, :]

        # Map reduced dimension and normalized inputs back to physical space
        x_inv = x_inv[:, :-xM_padding]
        cst, alpha = self.recover_full_cst(x_inv, af)

        return cst, alpha

    def hierarchical_design(self):
        # Function that will handle sampling from the various baseline models
        pass

    def recover_full_cst(self, x, af):
        N = x.shape[0]
        m = self.Ws[af].shape[0]
        x_shape, alpha = x[:, :-1].numpy(), x[:, -1:].numpy()

        # Uses a hit-and-run sampler to map reduced dimensional inputs back into full space
        cst = np.zeros((N, m))
        domain = psdr.BoxDomain(-1.01*np.ones(m), 1.01*np.ones(m))
        for i in range(N):
            con_domain = domain.add_constraints(A_eq=self.Ws[af].T, b_eq=x_shape[i, :])

            keep_trying, failed_attempts = True, 0
            while keep_trying:
                try:
                    cst_i = con_domain.sample(10)
                    keep_trying = False
                except:
                    failed_attempts += 1
                    if failed_attempts > 10:
                        assert False, 'Too many failed attempts'
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


