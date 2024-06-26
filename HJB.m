% Explicit solver para
% HJB ECUATION

clear;

paths = 100000; 
N_T = 500;

T = 1.0;
dt = T/N_T;
delta_W = sqrt(dt)*randn(paths, N_T-1); 

alpha = 0.2;
sigma = sqrt(2*alpha);
X_0 = 0;

X = zeros(paths, N_T);
X(:,1) = X_0;

X_next_EM = @(X_prev, dW, n_t) X_prev + sigma.*dW;

for t = 1:N_T-1 
    X(:,t+1) = X_next_EM(X(:,t), delta_W(:,t), t);
end

% Condiciones finales:
% Creación de procesos Y y Z
Y_T = log(0.5*(1+X(:,N_T).^2));
Y = zeros(1,N_T);
Y(N_T) = mean(Y_T);

Z_T = sigma*4*X(:,N_T)./(1+X(:,N_T).^2); 
Z = zeros(1,N_T);
Z(N_T) = mean(Z_T);

% Creación función f
f = @(y, z, x, n_t) 3.0/20.0 * y.*z - cos(n_t*dt+x).*(1+y);
% Hay que pasarle: x, y, z en el tiempo adecuado

for i = 2:N_T-1
    j = N_T-i; % j va de N_T-2 a 1 
    % Este esquema asume:  Theta_1 = 0   Theta_2 = 0.5   Theta_3 = 0.5 
    Z(1, j) = - mean(Z(:, j+1)) + 1.0/(dt*0.5) * mean(Y(:,j+1).*delta_W(:,j+1)) ...
              + mean(f(Y(:,j+1), Z(:,j+1), X(:, j+1), j+1) .* delta_W(:,j+1) );
    Y(1, j) = mean(Y(:, j+1)) + dt * mean(f( Y(:,j+1), Z(:, j+1), X(:, j+1), j+1));
end


Y_an = -0.154159; Y=Y+0.38;
fprintf('Y num %f\n Y an %f\n Error abs %e\n', Y(1), Y_an, sqrt((Y_an - Y(1))^2))