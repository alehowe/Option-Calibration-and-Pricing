
function F_sim = simulateMC(F0, dates, sigma_hat, Nsim)

    % Function to simulate the futures
    
    % Inputs:
    %   F0         - Forward price at time t=0 (scalar)
    %   dates      - time fractions between MC reset dates (vector)
    %   sigma_hat  - Adjusted volatility (vector, one for each reset date
    %                in (i-1, i))
    %   Nsim       - Number of MC simulations (scalar)
    %
    % Outputs:
    %   F_sim      - Simulated future (matrix)
    

    % Initialize the simulation matrix
    M = length(dates);
    F_sim = zeros(Nsim, M+1);
    
    % Set F0
    F_sim(:, 1) = F0;
    
    % Initialize the random variales Z
    Z = randn(Nsim, M);
    
    for jj=1:M
        
        % Compute the drift at the given reset date andd simulate
        F_sim(:, jj+1) = F_sim(:, jj).*exp(-0.5*sigma_hat(jj)^2 + Z(:, jj)*sigma_hat(jj));
    
    end

end



