# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:33:41 2024

@author: Miguel Sánchez
"""

"""
Valores con los que empezar:
    
N_0 = 100
N_b = 100
N_r = 1000

lr_1 = 0.01
lr_2 = 0.001
lr_3 = 0.0001
lr_salto_1 = 2000
lr_salto_2 = 4000

ep = 5000

n_h_l = 8
n_n_p_l = 15

act_fun = "tanh"
"""


import numpy as np
import tensorflow as tf

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Parámetros la ecuación
name = "American Option"

T = 3.0;
r = 0.08
y = 0.12
sigma = 0.2
k = 100
X_0 = 80

t_min = 0.
t_max = T
x_min = 0 # Hay que mirarlo en x=80
x_max = 160

# Tomar los límites
def get_info():
    return t_min, t_max, x_min, x_max, name

# Solución analítica
""" En la primera llamada ejecutar=False, cuando PINN compruebe que existe la
solución an, entunces volverá a llamar a la función  con ejecutar=True"""
def existe_an(ejecutar, t=None, x=None):
    #if ejecutar:
    #    return func_an(t, x)
    return False # Devolver True si existe la función analítica, en este caso False
#def func_an(t, x):
# NO EXISTE LA ANALÍTICA
#     return 

# Condición Final
def func_u_f(x):
    sol = np.zeros( (x.shape[0], 1) )
    for i in range(0, x.shape[0]):
        sol[i] = max(k-x[i], 0)
    return tf.constant(sol, dtype=DTYPE)

# Condición de contorno
def func_u_b(t, x):
    sol = np.zeros( (x.shape[0], 1) )
    for i in range(0, x.shape[0]):
        if x[i] == x_max:
            sol[i] = 0
        elif x[i] == x_min:
            sol[i] = k
        else:
            print("Error: El contorno debe ser 0 ó N")
    return tf.constant(sol, dtype=DTYPE)

# Residuo
def func_r(t, x, u, u_t, u_x, u_xx):
    return u_t + 0.5*sigma**2 * (x**2)*u_xx  + (r-y)*x*u_x - r*u + I1(x,u)*(r*k-y*x) 
def I1(x, v):
    comparison = tf.cast(k - x >= v, dtype=tf.float32)
    return comparison










