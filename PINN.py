# -*- coding: utf-8 -*-
"""
Created on Wed May  8 20:09:57 2024

@author: Miguel Sánchez

Modelo de red neuronal que resuelve PDEs numéricamente

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from AO import func_u_f, func_u_b, func_r, get_info, existe_an

"""***************************PARÁMETROS DE LA RED**************************"""
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


"""************************CREACIÓN DATASET DE LA RED************************"""
# Seleccionar el tipo de dato
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Semilla para obtener resultados reproducibles
tf.random.set_seed(0)

# Obtenemos los límites
tmin, tmax, xmin, xmax, name = get_info()
lb = tf.constant([tmin, xmin], dtype=DTYPE)
ub = tf.constant([tmax, xmax], dtype=DTYPE)

# Dataset en instante FINAL
t_f_tensor  = tf.ones((N_0,1), dtype=DTYPE)*ub[0]
x_f         = np.random.uniform(xmin, xmax, (N_0,1))
x_f_tensor  = tf.constant(x_f, dtype=DTYPE)

X_f         = tf.concat([t_f_tensor, x_f_tensor], axis=1)
u_f         = func_u_f(x_f)
"""
X_0: Tensor (matriz de tensorflow) N_0x2, donde la primera columna son todo
    los valores del t_min y la segunda son puntos aleatorios entre x_min y x_max
Únicamente con x_0 ya puedo calcular u_0
"""

# Dataset para condiciones contorno
t_b         = np.random.uniform(tmin, tmax, (N_b,1))
x_b         = np.random.choice([xmin, xmax], (N_b, 1))
t_b_tensor  = tf.constant(t_b, dtype=DTYPE)
x_b_tensor  = tf.constant(x_b, dtype=DTYPE)

X_b         = tf.concat([t_b_tensor, x_b_tensor], axis=1)
u_b         = func_u_b(t_b, x_b)
"""
X_b: Tensor N_bx2 donde la primera columna son tiempos entre t_min y t_max y la segunda
    columna son puntos aleatorios que siguen una distribución de bernoulli entre x_min y x_max
Bernoulli, en vez de generar N_b/2 en el límite inferior y N_b/2 en el superior, 
    genero N_b en total con un 50% de prob de ser en límite sup y 50% prob de
    ser en el límite inf
Con X_b calculo u_b
"""

# Dataset en puntos del interior
t_r_tensor = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
x_r_tensor = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([t_r_tensor, x_r_tensor], axis=1)

X_data = [X_f, X_b]
u_data = [u_f, u_b]
"""
X_r: Tensor N_rx2, Primera col tiempos aleatorios entre t_min y t_max
    Segunda col puntos aleatorios entre x_min y x_max
"""


"""
************************CREACIÓN DEL MODELO***********************************

Input: Hidden Layers, Neurons per layer
Output: Modelo de red neuronal que toma como input 2 valores (t, x), los
    escala al rango [lb, ub]. Como output devuelve un único valor
    Añade las capas y neuronas indicadas
OJO: 
    Función de activación: Tanh (cambiarla si hace falta)
    Glorot normal initialization Método para inicializar pesos para 
    evitar vanishing o exploding gradients durante el entrenamiento. Dejarlo así
"""
def init_model(num_hidden_layers=n_h_l, num_neurons_per_layer=n_n_p_l):
    
    # Iniciar modelo
    model = tf.keras.Sequential()

    # Input 
    model.add(tf.keras.Input(2))

    # Hacer el scaling
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    model.add(scaling_layer)

    # Añadir las hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer, activation=tf.keras.activations.get(act_fun),
            kernel_initializer='glorot_normal'))

    # Output
    model.add(tf.keras.layers.Dense(1))
    
    return model

"""
Función para calcular el residuo r. Ayudarse sí o sí de tf.GradientTape, si no es
un absoluto lío inviable de realizar a mano

Para calcular r necesario u_t, u_x y u_xx, cosa que hace con GradientTape

Ojo, esas derivadas son respecto al cálculo numérico de la red, es decir, no son
derivadas exactas de la solución exacta ni nada de eso. Lo que calculan es la
variación del resultado u de la red respecto a variaciones en t, x y xx

Tiene lógica que de esta forma pueda entrenarse la red. La red simula ser la función
u, entonces al crear la ecuación del residuo usando derivadas numéricas de la
propia red, si el residuo es 0, el valor numérico de u dado por la red debe ser 
el correcto
"""
def comp_r(model, X_r):
    
    # Gradient Tape para calcular derivadas TF
    with tf.GradientTape(persistent=True) as tape:
        # Recupera t y x de X_r
        t, x = X_r[:, 0:1], X_r[:,1:2]

        tape.watch(t)
        tape.watch(x)
 
        """ Calcular valor u dado por la red """
        u = model(tf.stack([t[:,0], x[:,0]], axis=1))

        u_x = tape.gradient(u, x)
            
    u_t = tape.gradient(u, t)
    u_xx = tape.gradient(u_x, x)

    del tape

    return func_r(t, x, u, u_t, u_x, u_xx)

"""
CÁLCULO DEL LOSS. Ahora se tiene en cuenta el residuo, pero también los errores
en los boundary y en CI

Phi = Phi^r + Phi^0 + Phi^b 
"""
def calc_loss(model, X_r, X_data, u_data):
    
    # Calcular phi^r
    r = comp_r(model, X_r)
    phi_r = tf.reduce_mean(tf.square(r))
    
    # Inicializar loss
    loss = phi_r
    
    # Añadir phi^0 y phi^b
    for i in range(0, len(X_data)): # Este bucle es un poco tontería, sólo son 2 variables, t=0 y x=Boundary
        u_pred = model(X_data[i])
        loss += tf.reduce_mean(tf.square(u_data[i] - u_pred))
    
    return loss

"""
Descenso del gradiente
"""
def comp_grad(model, X_r, X_data, u_data):
    
    with tf.GradientTape(persistent=True) as tape:
        """ Este tape es para las derivadas respecto a los pesos y biases para
        entrenar la red. Se obtiene el gradiente respecto a esas variables"""
        tape.watch(model.trainable_variables) # Eso son todos los pesos y biases
        loss = calc_loss(model, X_r, X_data, u_data)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g

"""
******************************************************************************
Inicialización del modelo y resolución de la PDE
******************************************************************************
"""
model = init_model()
"""
Learning Rate variante 
Iter [0 - lr_salto_1]           lr_1
Iter [lr_salto_1 - lr_salto_2]  lr_2
Iter [lr_salto_2 - onwa]        lr_3
"""
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([lr_salto_1,lr_salto_2],[lr_1,lr_2,lr_3])

# Elegir el Adam que según Andrew NG es el mejor
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Definir training step como función de TF para aligerar la ejecución
@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = comp_grad(model, X_r, X_data, u_data)
    
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    
    return loss

histo = []
t_0 = time()

for i in range(ep+1):
    
    loss = train_step() # Sin minibatches (no son tan recomendados en PINNs)
    
    # Append current loss to hist
    histo.append(loss.numpy())
    
    # Output current loss after 50 iterates
    if i%100 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i,loss))

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t_0))

"""
******************************************************************************
Plotear la solución y la evolución del loss:
******************************************************************************
"""

N_grid_repr = 600
tspace = np.linspace(lb[0], ub[0], N_grid_repr + 1)
xspace = np.linspace(lb[1], ub[1], N_grid_repr + 1)
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T

# Predicción:
upred = model(tf.cast(Xgrid,DTYPE))

U = upred.numpy().reshape(N_grid_repr+1,N_grid_repr+1)
"""
U cambia la forma de u_ex a una matriz NxN de la forma
t_1 x_1,    t_2 x_1,    t_3 x_1 ...
t_1 x_2,    t_2 x_2,    t_3 x_2 ...
.
.
t_1 x_n,    t_2 x_n,    t_3 x_n ...
"""

# Surface plot of solution u(t,x)
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, U, cmap='viridis');
ax.view_init(40,35)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$u_\\theta(t,x)$')
ax.set_title(name+' PINN Solution');
plt.savefig(name+f'PINN_{loss}.pdf', bbox_inches='tight', dpi=300)

# Plot del error
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(histo)), histo,'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$');
plt.savefig(name+f'_Error_{loss}.pdf', bbox_inches='tight', dpi=300)


""" 
******************************************************************************
IMPRIMIR PARÁMETROS EN UN FICHERO PARA MEJORARLOS
******************************************************************************
"""
def create_loss_file(loss, data):
    filename = name+f"_loss_{loss}.txt"
    with open(filename, 'w') as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")

data = {
    "Loss": loss,
    "N_0": N_0,
    "N_b": N_b,
    "N_r": N_r,
    "lr_1": lr_1,
    "lr_2": lr_2,
    "lr_3": lr_3,
    "lr_salto_1": lr_salto_1,
    "lr_salto_2": lr_salto_2,
    "ep": ep,
    "n_h_l": n_h_l,
    "n_n_p_l": n_n_p_l,
    "act_fun": act_fun
}

create_loss_file(loss, data)

""" 
******************************************************************************
PLOTEAR LA SOLUCIÓN ANALÍTICA SI EXISTE
******************************************************************************
"""
if existe_an(False):
    an_sol = existe_an(True, T, X)
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, an_sol, cmap='viridis');
    ax.view_init(40,35)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u_\\theta(t,x)$')
    ax.set_title(name+' Analyt Solution');














