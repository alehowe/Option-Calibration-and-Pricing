function [Prices, sigma_hat] = calibration_single_const(sigma_1, sigma_2, F0, Strikes, TTM, discounts)
    % Calibration of prices and implied volatilities using a constant volatility 
    % in one case and a time dependent one in the other one.
    % In order to be generic, we are modeling the integral in this case, not
    % the implied volatilities value.
    
    % INPUTS:
    % sigma_1    - Array of incremental volatilities for the first component
    % sigma_2   - Constant volatility for the second component
    % F0        - Initial forward price
    % Strikes   - Array of strike prices
    % TTM       - Time to maturity (in years)
    % discounts - Discount factors
    %
    % OUTPUTS:
    % Prices    - Calculated option prices based on the Black model
    % sigma_hat - Implied volatilities computed for the model
    
    sigma_hat = sqrt(cumsum(sigma_1) + sigma_2^2.*TTM);

    Prices = zeros(length(TTM), length(Strikes));

    for i = 1:length(TTM)
        for j = 1:length(Strikes)
            % Compute the option price using the Black formula
            Prices(i, j) = blackformula(F0, Strikes(j), discounts(i), sigma_hat(i));
        end
    end

    sigma_hat = repmat(sigma_hat, 1, length(Strikes));
end