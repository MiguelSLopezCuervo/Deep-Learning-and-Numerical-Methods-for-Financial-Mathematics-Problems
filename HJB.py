# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:54:25 2024

@author: Miguel Sánchez
"""

"""
POSIBLES PARÁMETROS CON LOS QUE EMPEZAR A TESTEAR: ver más en la carpeta específica

N_0 = 100 # Puede ser N_F según el problema, pero por simplicidad se denota siempre como N_0
N_b = 100
N_r = 200

lr_1 = 2.5e-3
lr_2 = 2.5e-4
lr_3 = 1e-5
lr_salto_1 = 5000
lr_salto_2 = 12000

ep = 15000

n_h_l = 30
n_n_p_l = 15

act_fun = 'tanh'
"""

import numpy as np
import tensorflow as tf

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Parámetros la ecuación
name = "A HJB Equation"

alpha = .2 # (0, inf)
beta = -1. # R \ {0}
paths = 10**4
n_T = 100  
n_X = 200 

t_min = 0.
t_max = 1.
x_min = -1.
x_max = 1.

# Tomar los límites
def get_info():
    return t_min, t_max, x_min, x_max, name

""" 
******************************************************************************
GENERACIÓN DE LOS PUNTOS QUE SE INTERPOLARÁN PARA LA SOLUCIÓN ANALÍTICA 
******************************************************************************
"""
# Condición terminal (y función necesaria para el cálculo analítico en todo t)
def g(x):
    return np.log(0.5*(1+x**2))

dt = (t_max-t_min)/n_T
dx = (x_max-x_min)/n_X

tspace = np.linspace(t_min, t_max, n_T)
xspace = np.linspace(x_min, x_max, n_X)

delta_W = np.sqrt(dt) * np.random.randn(paths, n_T-1)
W = np.cumsum(delta_W, axis=1)
W = np.insert(W, 0, 0, axis=1)
def u_an_fij(n_t, x):
    res = np.exp( beta/alpha * g( x + np.sqrt(2*alpha)*W[:,n_T-1-n_t] ) )
    res = np.mean(res)
    return alpha/beta * np.log( res )
u_fijo = np.zeros(( (n_T)*(n_X),1 ))
for i in range(0, n_X):
    for j in range(0, n_T):
        u_fijo[i*n_T+j] = u_an_fij(j, xspace[i])
"""
u_ex es un vector de dimensión (n_T*n_X,1). Sus componentes van según:
(t_1 x_1, t_2 x_1, t_3 x_1 ... t_N x_1, t_1 x_2, t_2 x_2, t_3 x_2 ... t_N x_N)
"""
U_ex_fijo = u_fijo.reshape(n_X,n_T)





""" En la primera llamada ejecutar=False, cuando PINN compruebe que existe la
solución an, entunces volverá a llamar a la función  con ejecutar=True"""
def existe_an(ejecutar, t=None, x=None):
    if ejecutar:
        return func_an(t, x)
    return True # Devolver True si existe la función analítica, en este caso True
"""
func_an debe hacer una media de los puntos que sí conocemos. Se procede así:
1º Se calcula la p_ex = posición exacta en la que se encuentra el punto 
    cuya solución se debe calcular
2º Se calculan las 4 posiciones exactas de U_ex_fijo que encierran a p_ex
3º Se hace una media ponderada de esas 4 posiciones

IMPORTANTE: NOTACIÓN: 
Los diferenciales con barrabaja, d_x, d_t corresponden
a las diferencias de los puntos que llegan a la función
Los diferenciales sin barrabaja, dx, dt, corresponden
a las diferencias de los puntos fijos
"""
def func_an(t, x):
    res = np.zeros( (x.shape[0], t.shape[1]) )
    for i in range(0, x.shape[0]):
        for j in range(0, t.shape[1]):
            pos_x = x[i, 0]
            pos_t = t[0, j]
            
            N_x_i  = int( (pos_x-x_min)/(x_max-x_min) * n_X )
            N_x_d  = int( (pos_x-x_min)/(x_max-x_min) * n_X ) + 1
            N_t_ab = int( (pos_t-t_min)/(t_max-t_min) * n_T )
            N_t_ar = int( (pos_t-t_min)/(t_max-t_min) * n_T ) + 1
            
            dist_x_i  = pos_x - (N_x_i *dx+x_min)
            dist_t_ab = pos_t - (N_t_ab*dt+t_min)
            
            while N_x_d >= n_X:
                N_x_i-=1
                N_x_d-=1
                dist_x_i=1
            while N_t_ar >= n_T:
                N_t_ab-=1
                N_t_ar-=1
                dist_t_ab=1
            
            eq_t_ab = (1-dist_x_i)*U_ex_fijo[N_x_i,N_t_ab] + dist_x_i*U_ex_fijo[N_x_d,N_t_ab]
            eq_t_ar = (1-dist_x_i)*U_ex_fijo[N_x_i,N_t_ar] + dist_x_i*U_ex_fijo[N_x_d,N_t_ar]
            
            res[i, j] = (1-dist_t_ab)*eq_t_ab + dist_t_ab*eq_t_ar
            
    return res

# Condición inicial
def func_u_f(x):
    N = x.shape[0]
    tspace = np.linspace(t_min, t_max, N)
    T, X = np.meshgrid(tspace, x)
    sol = func_an(T,X)
    sol = sol[:,-1]
    return tf.constant(sol, dtype=DTYPE)

# Condición de contorno
def func_u_b(t, x):
    T, X = np.meshgrid(t, x)
    sol = func_an(T,X)
    sol = sol[0,:]
    return tf.constant(sol, dtype=DTYPE)

# Residuo
def func_r(t, x, u, u_t, u_x, u_xx):
    return u_t + alpha * u_xx + beta * u_x*u_x













