# -*- coding: utf-8 -*-
"""
Created on Wed May  8 20:16:22 2024

@author: Miguel Sánchez

Reaction-Diffusion Equation
"""

"""
Aprox probar:
    
N_0 = 50 
N_b = 50 
N_r = 10000

lr_1 = 1e-2
lr_2 = 5e-3
lr_3 = 1e-3
lr_salto_1 = 250
lr_salto_2 = 800

ep = 1000

n_h_l = 20
n_n_p_l = 10

act_fun = 'tanh'

"""

import numpy as np
import tensorflow as tf

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Parámetros la ecuación
name = "Reaction–Diffusion"

sigma = .1
k = 6/10
lambd = 1

t_min = 0.
t_max = 1.
x_min = -5.
x_max = 5.

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
    res = 1+k+np.sin( lambd*x )*np.exp( lambd**2*( t-t_max )/2 )
    return res

# Condición inicial
"""
SE PUEDE PARALELIZAR
"""
def func_u_f(x):
    sol = np.zeros( (x.shape[0], 1) )
    for i in range(0, x.shape[0]):
        sol[i] = 1+k+np.sin( lambd*x[i] )
    return tf.constant(sol, dtype=DTYPE)

# Condición de contorno
def func_u_b(t, x):
    sol = np.zeros( (x.shape[0], 1) )
    sol =  1 + k + np.sin( lambd*x )*np.exp( lambd**2*(t-t_max )/2 )
    return tf.constant(sol, dtype=DTYPE)

# Residuo
def func_r(t, x, u, u_t, u_x, u_xx):
    return u_t + 0.5*sigma**2 + tf.minimum(1.0, (u - k - 1 - tf.math.sin(lambd*x) * tf.math.exp(lambd**2 * (t - t_max) / 2))**2)









