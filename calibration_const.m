function [Prices, sigma_hat] = calibration_const(sigma_1, sigma_2, F0, Strikes, TTM, discounts)
    % This function calibrates constant volatility parameters for the Black model
    % and computes option prices based on the calibrated volatilities.

    % Inputs:
    % sigma_1, sigma_2 - Constant volatility parameters
    % F0 - Forward price at time t=0
    % Strikes - Array of strike prices
    % TTM - Time to maturity (vector)
    % discounts - Discount factors for the corresponding maturities

    % Outputs:
    % Prices - Computed option prices using the Black model
    % sigma_hat - Calibrated volatilities (per maturity)
    
    sigma_hat = sqrt(sigma_1^2.*TTM + sigma_2^2.*TTM); % Total calibrated volatility

    Prices = zeros(length(TTM), length(Strikes));

    for i = 1:length(TTM)
        for j = 1:length(Strikes)
            % Compute the option price using the Black formula
            Prices(i, j) = blackformula(F0, Strikes(j), discounts(i), sigma_hat(i));
        end
    end

    % Replicate sigma_hat across all strikes for compatibility with Prices
    sigma_hat = repmat(sigma_hat, 1, size(Prices, 2));

end
