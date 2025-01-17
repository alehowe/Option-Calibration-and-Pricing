function [Prices, sigma_hat] = calibration_time_dep(sigma1, sigma2, F0, Strikes, TTM, discounts)
    
    % Function to compute prices and volatility (sigma_hat) in a time-dependent calibration model
    
    % Inputs:
    %   sigma1     - Time-dependent volatility component 1 (vector)
    %   sigma2     - Time-dependent volatility component 2 (vector)
    %   F0         - Forward price at time t=0 (scalar)
    %   Strikes    - Strike prices (vector)
    %   TTM        - Time to maturity (vector)
    %   discounts  - Discount factors (vector)
    %
    % Outputs:
    %   Prices     - Computed option prices (matrix)
    %   sigma_hat  - Adjusted volatility matrix for pricing (matrix)

    sigma_hat = sqrt(cumsum(sigma1) + cumsum(sigma2));

    Prices = zeros(length(TTM), length(Strikes));

    for i = 1:length(TTM)
        for j = 1:length(Strikes)
            % Compute the option price using the Black formula
            Prices(i, j) = blackformula(F0, Strikes(j), discounts(i), sigma_hat(i));
        end
    end
    
    sigma_hat = repmat(sigma_hat, 1, length(Strikes));

end