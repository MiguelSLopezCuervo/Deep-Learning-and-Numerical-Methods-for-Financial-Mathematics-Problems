% Explicit solver para Cálculo del precio justo de
% American Put Option
clear;

paths = 100000; 
N_T = 6000;

T = 3.0;
dt = T/N_T;
delta_W = sqrt(dt)*randn(paths, N_T-1); 
r = 0.08;
y = 0.12;
sigma = 0.2;
k = 100;
X_0 = 80;

X = zeros(paths, N_T);
X(:,1) = X_0;

X_next_EM = @(X_prev, dW, n_t) ...
            X_prev + (r-y)*X_prev * dt + sigma*X_prev.*dW;
% Con Milstein u otro se ganaría precisión en esta simulación
for t = 1:N_T-1 
    X(:,t+1) = X_next_EM(X(:,t), delta_W(:,t), t);
end

% %Graficar los X
% figure; 
% plot((1:size(X,2)) * dt, X')
% xlabel('Time'); 
% ylabel('Value of X_t'); 
% title('Stochastic Processes X'); 
% %Graficar la media de los X
% figure; 
% plot((1:size(X,2)) * dt, mean(X', 2)) % más sencillo plot((1:size(X,2)) * dt,mean(X))
% xlabel('Time'); 
% ylabel('Value of X_t'); 
% title('Stochastic Process X, mean'); 

% Condiciones finales:
V_F = @(x, k) max(k-x, 0);
Z_F = @(x, k) -sigma * (x >= k);

% Creación de procesos V y Z
V_T = V_F( X(:, N_T), k);
V = zeros(1,N_T);
V(N_T) = mean(V_T);

Z_T = Z_F(X(:, N_T), k);
Z = zeros(1,N_T);
Z(N_T) = mean(Z_T);

% Creación función f
f = @(v, z, x)  -( r*( v - k*I1(x, v, k) ) + y*x.*I1(x, v, k) );
f_escs = @(v, z, x)  -( r*( v - k*I1(x, v, k) ) + y*x.*I1(x, v, k) );
% Hay que pasarle: x, v, z en el tiempo adecuado

% Primera iteración
Z(end-1) = - Z(end) + 1.0/(dt*0.5) * mean(V_T.*delta_W(:,end)) ...
              + mean( f(V_T, Z_T, X(:, end)).*delta_W(:,end) );
V(end-1) = V(end) + dt * mean( f( V_T, Z_T, X(:, end)) );
% Bucle
for i = 2:N_T-1
    j = N_T-i; % j va de N_T-2 a 1 
    % Este esquema asume:  Theta_1 = 0   Theta_2 = 0.5   Theta_3 = 0.5
    Z(j) = - Z(j+1) + 1.0/(dt*0.5) * mean( V(j+1)*delta_W(:,j) ) ...
           + mean( f_escs( V(j+1), Z(j+1), X(:, j+1) ) .* delta_W(:,j) );
    V(j) = V(j+1) + dt * mean( f_escs( V(j+1), Z(j+1), X(:, j+1) ) );
    
end

% Analytical solution: Obtenida del paper
% Calcular el error
V_an = 25.65; 
fprintf('V num %f\n V an %f\n Error cuadrático %e\n', ...
    mean(V(:,1)), V_an, (V_an - mean(V(:,1)))^2 )

% Función para calcular los I
function resultado = I1(x, v, k)
    resultado = (-x+k >= v);
end