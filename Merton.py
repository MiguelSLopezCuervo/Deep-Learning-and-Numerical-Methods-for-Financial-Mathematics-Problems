# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:06:12 2024

@author: Miguel Sánchez
"""

"""
HASTA AHORA NINGUNA COMBINACIÓN DE HPPARÁMETROS HA DADO BUENOS RESULTADOS.
PROBAR NUEVAS
"""


import numpy as np
import tensorflow as tf

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Parámetros la ecuación
name = "Merton Problem"

r = 0.2
R = 0.3
sigma = 0.15
lamb = R-r

t_min = 0.
t_max = 1.
x_min = 1.
x_max = 160.

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
    sol = np.zeros( (x.shape[0], t.shape[1]) )
    for i in range(0, x.shape[0]):
        if t[0,i] == t_max:
            sol[:,i] = 0
        else:
            sol[:,i] = (t_max-t[:,i])*np.log( (x[:,i])/(t_max-t[:,i]) ) + \
                0.5 * ( r+0.5*(lamb/sigma)**2 ) * (t_max-t[:,i])**2
    return sol

# Condición inicial
def func_u_f(x):
    sol = np.zeros( (x.shape[0], 1) )
    return tf.constant(sol, dtype=DTYPE)

# Condición de contorno
def func_u_b(t, x):
    sol = np.zeros( (x.shape[0], 1) )
    sol =  (t_max-t)*np.log( (x)/(t_max-t) ) + \
            0.5 * ( r+0.5*(lamb/sigma)**2 ) * (t_max-t)**2
    return tf.constant(sol, dtype=DTYPE)

# Residuo
def func_r(t, x, u, u_t, u_x, u_xx):
    #return u_t - (0.5*lamb**2/sigma**2) * (u_x**2/u_xx) + r*x*u_x - ( 1+tf.math.log(u_x) )
    # Possible simplification that makes the problem simpler (but still unsolvable)
    return u_t - (0.5*lamb**2/sigma**2) * (u_x**2/u_xx) + r*x*u_x - ( 1+tf.math.log((t_max-t)/x) )
