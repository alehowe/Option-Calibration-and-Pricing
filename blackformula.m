function Price = blackformula(F0, K, discount, sigma_hat)
    % blackformula computes the price of an option using the Black formula
    % 
    % Inputs:
    %   F0         - Forward price at time 0
    %   K          - Strike price of the option
    %   discount  - Discount factor for present value calculation
    %   sigma_hat  - Implied volatility of the option
    %
    % Output:
    %   Price      - Calculated price of the option

    d2 = (log(F0/K) - 0.5 * sigma_hat^2) / (sigma_hat);
    d1 = d2 + sigma_hat;
    
    % Compute the option price using the Black formula
    Price = discount * (F0 * normcdf(d1) - K * normcdf(d2));
end