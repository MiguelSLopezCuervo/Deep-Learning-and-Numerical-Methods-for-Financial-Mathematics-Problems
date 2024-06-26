# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:53:10 2024

@author: Miguel Sánchez
"""

"""
RESULTADOS MUY BUENOS, EMPEZAR A TUNEAR POR AQUÍ:

N_0 = 100 
N_b = 100
N_r = 10000

lr_1 = 1e-2
lr_2 = 5e-2
lr_3 = 1e-3
lr_salto_1 = 1000
lr_salto_2 = 10000

ep = 500

n_h_l = 10
n_n_p_l = 15

act_fun = 'tanh'
"""

import numpy as np
import tensorflow as tf

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Parámetros la ecuación
name = "A FBSDE from Zhao"


t_min = 0.
t_max = 1.
x_min = -2
x_max = 2

# Tomar los límites
def get_info():
    return t_min, t_max, x_min, x_max, name

# Solución analítica
""" En la primera llamada ejecutar=False, cuando PINN compruebe que existe la
solución an, entunces volverá a llamar a la función  con ejecutar=True"""
def existe_an(ejecutar, t=None, x=None):
    if ejecutar:
        return func_an(t, x)
    return True # Devolver True si existe la función analítica, en este caso True
def func_an(t, x):
    res = np.sin(t+x)
    return res

# Condición inicial
def func_u_f(x):
    sol = np.zeros( (x.shape[0], 1) )
    sol = np.sin(x+1)
    return tf.constant(sol, dtype=DTYPE)

# Condición de contorno
def func_u_b(t, x):
    sol = np.zeros( (x.shape[0], 1) )
    sol =  np.sin(t+x)
    return tf.constant(sol, dtype=DTYPE)

# Residuo
def func_r(t, x, u, u_t, u_x, u_xx):
    return u_t + tf.math.sin(t+x)*u_x + 9/200*tf.math.cos(t+x)*tf.math.cos(t+x)*u_xx + \
        tf.math.cos(t+x) * ( 9/200*u*u_x-(1+u) )
