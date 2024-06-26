clear;

T = 1;
dt = 0.001;
N_T = T/dt;
paths = 100000;
delta_W = sqrt(dt)*randn(paths, N_T); 

X_0 = 1;
X = zeros(paths, N_T);
X(:,1) = X_0;

X_next_EM = @(X_prev, W_prev, n_t) ...
            X_prev + sin((n_t-1)*dt + X_prev)*dt + ...
            3.0/10.0 * cos(n_t*dt-1*dt + X_prev).*W_prev;

for t = 1:N_T-1 
    X(:,t+1) = X_next_EM(X(:,t), delta_W(:,t), t);
end

% Creación de procesos Y y Z
Y_T = sin(T+X(:, N_T));
Y = zeros(paths, N_T);
Y(:,N_T) = Y_T;

Z_T = 3.0/10.0 * cos(T+X(:, N_T)).^2;
Z = zeros(paths, N_T);
Z(:,N_T) = Z_T;

f = @(y, z, x, n_t) 3.0/20.0 * y.*z - cos(n_t*dt+x).*(1+y);

for i = 1:N_T-1
    j = N_T-i; % j va de N_T-1 a 1 
    % Este esquema asume:  Theta_1 = 0   Theta_2 = 0.5   Theta_3 = 0.5 
    Z(:, j) = - mean(Z(:, j+1)) + 1.0/(dt*0.5) * mean(Y(:,j+1).*delta_W(:,j+1)) ...
              + mean(f(Y(:,j+1), Z(:,j+1), X(:, j+1), j+1) .* delta_W(:,j+1) );
    Y(:, j) = mean(Y(:, j+1)) + dt * mean(f( Y(:,j+1), Z(:, j+1), X(:, j+1), j+1));
end

% Analytical solution
Y_ex = @(t, x) sin(t + x);

% Calcular el error
Y_an = Y_ex( 0, 1 );

fprintf('Y num %f\n Y an %f\n Error cuadrático %e\n', ...
    mean(Y(:,1)), Y_an, (Y_an - mean(Y(:,1)))^2 )

