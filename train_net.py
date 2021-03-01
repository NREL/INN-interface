import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from inv_net import InvNet
from INN import INN
from time import strftime
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

dim_red = 'red'
n_layers = 15

learning_rate = 1e-4

load_model_path = None
save_model_path = '/'.join(['model', 'trainingExample_'+strftime('%Y%m%d-%H%M%S')])

df_train = pd.read_csv('data/DU25_A17-train.csv', index_col=0)
df_test = pd.read_csv('data/DU25_A17-test.csv', index_col=0)
#df_train = pd.read_csv('data/DU25_A17-train.csv', index_col=0)
#df_test = pd.read_csv('data/DU25_A17-test.csv', index_col=0)

cst_params_train = df_train[[str(i) for i in range(20)]].to_numpy()
trailing_edge_train = cst_params_train[:, -2:]
if np.all(np.min(trailing_edge_train, axis=0) == np.max(trailing_edge_train, axis=0)):
    cst_params_train = cst_params_train[:, :-2]

cst_params_test = df_test[[str(i) for i in range(20)]].to_numpy()
trailing_edge_test = cst_params_test[:, -2:]
if np.all(np.min(trailing_edge_test, axis=0) == np.max(trailing_edge_test, axis=0)):
    cst_params_test = cst_params_test[:, :-2]

thickness_train = df_train['t/c max'].to_numpy().reshape((-1, 1))
thickness_test = df_test['t/c max'].to_numpy().reshape((-1, 1))

Re_train = df_train['rey'].to_numpy().reshape((-1, 1))
Re_test = df_test['rey'].to_numpy().reshape((-1, 1))

alpha_train = df_train['aoa'].to_numpy().reshape((-1, 1))
alpha_test = df_test['aoa'].to_numpy().reshape((-1, 1))

stall_train = df_train['stall_margin'].to_numpy().reshape((-1, 1))
stall_test = df_test['stall_margin'].to_numpy().reshape((-1, 1))

cl_train = df_train['cl'].to_numpy().reshape((-1, 1))
cl_test = df_test['cl'].to_numpy().reshape((-1, 1))

cd_train = np.log10(df_train['cd'].to_numpy().reshape((-1, 1)))
cd_test = np.log10(df_test['cd'].to_numpy().reshape((-1, 1)))

cld_train = df_train['l/d'].to_numpy().reshape((-1, 1))
cld_test = df_test['l/d'].to_numpy().reshape((-1, 1))

if dim_red == 'full':
    W = np.eye(cst_params_train.shape[1])
elif dim_red == 'red':
    W = np.load('model/'+af+'/U.npy')
else:
    assert False

cst_params_train, sf_cst = norm_data(cst_params_train, scale_type='minmax')
cst_params_test, _ = norm_data(cst_params_test, sf_cst, scale_type='minmax')

alpha_train, sf_alpha = norm_data(alpha_train, scale_type='minmax')
alpha_test, _ = norm_data(alpha_test, sf_alpha, scale_type='minmax')

x_train = np.concatenate((cst_params_train@W, alpha_train), axis=1)
x_test = np.concatenate((cst_params_test@W, alpha_test), axis=1)
y_train = Re_train
y_test = Re_test
c_train = thickness_train
c_test = thickness_test
f_train = np.concatenate((cd_train, cld_train, stall_train), axis=1)
f_test = np.concatenate((cd_test, cld_test, stall_test), axis=1)
l_train = cl_train
l_test = cl_test

N_train, xM = x_train.shape
N_test = x_test.shape[0]
yM, cM, fM = y_train.shape[1], c_train.shape[1], f_train.shape[1]
lM = cl_train.shape[1]
if (xM+yM) < (yM+cM+fM+1):
    x_train = np.concatenate((x_train, np.zeros((N_train, cM+fM+1-xM))), axis=1)
    x_test = np.concatenate((x_test, np.zeros((N_test, cM+fM+1-xM))), axis=1)
    xM = x_train.shape[1]
if (xM+yM)%2 == 1:
    x_train = np.concatenate((x_train, np.zeros((N_train, 1))), axis=1)
    x_test = np.concatenate((x_test, np.zeros((N_test, 1))), axis=1)
    xM += 1
zM = xM - (cM+fM)

sf_x = [np.concatenate((sf_cst[0], sf_alpha[0]), axis=0), 
        np.concatenate((sf_cst[1], sf_alpha[1]), axis=0), 'minmax']
#x_train, sf_x = norm_data(x_train, scale_type='minmax')
#x_test, _ = norm_data(x_test, scale_factor=sf_x, scale_type='minmax')
y_train, sf_y = norm_data(y_train)
y_test, _ = norm_data(y_test, scale_factor=sf_y)
c_train, sf_c = norm_data(c_train)
c_test, _ = norm_data(c_test, scale_factor=sf_c)
f_train, sf_f = norm_data(f_train)
f_test, _ = norm_data(f_test, scale_factor=sf_f)
l_train, sf_l = norm_data(l_train)
l_test, _ = norm_data(l_test, scale_factor=sf_l)

scale_factors = {'x': sf_x, 'y': sf_y, 'c': sf_c, 'f': sf_f, 'l': sf_l, 
                 'trailing_edge': [np.mean(trailing_edge_train[:, 0]), np.mean(trailing_edge_train[:, 1]), 'minmax']}



x_in = tf.keras.Input(shape=(xM,))
y_in = tf.keras.Input(shape=(yM,))
c_in = tf.keras.Input(shape=(cM,))
f_in = tf.keras.Input(shape=(fM,))
z_in = tf.keras.Input(shape=(zM,))
l_in = tf.keras.Input(shape=(lM,))

train_data = (x_train, y_train, c_train, f_train, l_train)
test_data  = (x_test,  y_test,  c_test,  f_test, l_test)

train_ds = tf.data.Dataset.from_tensor_slices(train_data).shuffle(100000).batch(50)
test_ds = tf.data.Dataset.from_tensor_slices(test_data).batch(50)

permute_layers = tf.stack([tf.random.shuffle(tf.range(xM+yM)) for i in range(n_layers)])

model = InvNet(x_in, y_in, c_in, f_in, z_in, l_in,
               n_layers=n_layers,
               permute_layers=permute_layers,
               input_shape=tf.TensorShape([xM+yM]),
               W=W,
               scale_factors=scale_factors,
               model_path=load_model_path)

model.fit(train_ds, test_ds, learning_rate=learning_rate, N_reps=50,
          epochs=1000, print_every=1, save_every=2, save_model_path=save_model_path)




