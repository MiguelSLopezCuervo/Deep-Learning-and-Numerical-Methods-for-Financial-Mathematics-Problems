% Explicit solver para Reaction Difusion Eq en formulación de FBSDE
clear;

paths = 100000;
N_T = 100;

T = 1;
dt = T/N_T;
delta_W = sqrt(dt)*randn(paths, N_T-1); 

X_0 = 0; 
sigma = .1; 
k = 6/10;
lambd = 1;

X = zeros(paths, N_T);
X(:,1) = X_0;

X_next_EM = @(X_prev, dW, n_t) X_prev + sigma*dW;
for t = 1:N_T-1 
    X(:,t+1) = X_next_EM(X(:,t), delta_W(:,t), t);
end

% Condiciones finales:
Y_F = @(x, t) 1 + k + sin(lambd*x);
Z_F = @(x, t) lambd*cos(lambd*x);

% Creación de procesos Y y Z
Y_T = Y_F( X(:, N_T), t);
Y = zeros(1,N_T);
Y(N_T) = mean(Y_T);

Z_T = Z_F(X(:, N_T), t);
Z = zeros(1,N_T);
Z(N_T) = mean(Z_T);

% Creación función f
f = @(y, z, x, t) min( 1, [ y-k-1- sin( lambd*x )*exp( lambd^2*( t-T )/2 ) ].^2 ) ;
% Hay que pasarle: x, v, z en el tiempo adecuado

Z(end-1) = - Z(end) + 1.0/(dt*0.5) * mean(Y_T.*delta_W(:,end)) ...
              + mean( f(Y_T, Z_T, X(:, end), (N_T)*dt).*delta_W(:,end) );
Y(end-1) = Y(end) + dt * mean( f( Y_T, Z_T, X(:, end), (N_T)*dt ) );

for i = 2:N_T-1
    j = N_T-i; % j va de N_T-2 a 1 
    % Este esquema asume:  Theta_1 = 0   Theta_2 = 0.5   Theta_3 = 0.5 
    %Z(j) = - Z(j+1) + 1.0/(dt*0.5) * mean( Y(j+1)*delta_W(:,j) ) ...
    %      + mean( f( Y(j+1), Z(j+1), X(:, j+1), (j+1)*dt ) .* delta_W(:,j) );
    Y(j) = Y(j+1) + dt * mean( f( Y(j+1), Z(j+1), X(:, j+1), (j+1)*dt ) );
end

% Solución Analítica
Y_an = 1+k+sin(lambd*0)*exp(lambd^2*(-T)/2);

fprintf('V num %f\n V an %f\n Error abs %e\n', ...
    mean(Y(:,1)), Y_an, sqrt((Y_an - mean(Y(1)))^2) )
