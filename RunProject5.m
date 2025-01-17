% HJM Model Calibration and Pricing
clc
clear all
close all

%% MATLAB code to import data from an Excel file

tic

% File name of the Excel file
filename = 'DATA_DEEEX.xlsx';

% Read data from different sheets of the Excel file
% Read data from the specified sheet
prices_data = readtable(filename, 'Sheet', 'Prices');
options_data = readtable(filename, 'Sheet', 'OptionsOnQ42025');
discounts_data = readtable(filename, 'Sheet', 'discounts');

% Extract relevant data
expiry_dates = datetime(prices_data.("ExpiryDate"), 'Format', 'dd-MM-yyyy');
forward_rates = prices_data.Last;

% Discount factors
discount_dates = datetime(discounts_data{1, :}, 'ConvertFrom', 'excel');
discount_factors = discounts_data{2, :};

disp('Forward Rates:');
disp(table(expiry_dates, forward_rates));

%% Inizialization
F0 = table2array(prices_data(11,3)); % Forward in t0
discounts = table2array(discounts_data(2,:)); 
discounts_dates = table2array(discounts_data(1,:)); % Dates related to discounts
Strikes = table2array(options_data(1,2:end));
TTM_call = table2array(options_data(2:end,1)); 
mkt_vol = table2array(options_data(2:end,2:end));

disp('Discount Factors:');
disp(table(datetime(discounts_dates', 'ConvertFrom', 'excel'), discounts', ...
    'VariableNames', {'Date', 'DiscountFactor'}));

disp('Implied Volatilities:');
disp(table(TTM_call, mkt_vol));

% Discounts related to Time to Maturity
dates = datetime(discounts_dates, 'ConvertFrom', 'excel');
today = datetime(2024, 11, 4);
discounts_frac = yearfrac(today,dates,3);
zero_rates = -log(discounts)./discounts_frac; % Zero rates computation
zero_rates_call = interp1(discounts_frac,zero_rates,TTM_call);
discounts_call = exp(-zero_rates_call.*TTM_call); % Discount computation

% Market prices obtained with the Black Model
mkt_prices = zeros(length(TTM_call), length(Strikes));
for i = 1:length(TTM_call)
    for j = 1:length(Strikes)
        mkt_prices(i,j) = blkprice(F0, Strikes(j), -log(discounts_call(i))/TTM_call(i), TTM_call(i), mkt_vol(i,j));
    end
end

%% Plot
figure()
surf(Strikes, TTM_call, mkt_vol);

xlabel('Strikes');
ylabel('Time to Maturity (TTM)');
zlabel('Volatilitites');

figure()
surf(Strikes, TTM_call, mkt_prices);

xlabel('Strikes');
ylabel('Time to Maturity (TTM)');
zlabel('Prices');

figure()
plot(TTM_call,discounts_call);
xlabel('Time to Maturity (TTM)');
ylabel('Discounts');

%% Cutoff
K_cut = 550;
index = Strikes < K_cut;
Strikes = Strikes(index); 
mkt_vol = mkt_vol(:,index);
mkt_prices = mkt_prices(:,index);
max_vol = max(mkt_vol, [], "all"); 

%% Point 3
fun = @(x) mean((mkt_prices - calibration_const(x(1), x(2), F0, Strikes, TTM_call, discounts_call)).^2, "all");

lb = [0, 0]; % lower bounds
x0 = max_vol*ones(2,1);

% Increase tolerance and maximum iterations
options = optimoptions('fmincon', ...
    'MaxIterations', 100000, ...         % Increase maximum iterations
    'MaxFunctionEvaluations', 100000, ... % Increase function evaluations
    'OptimalityTolerance', 1e-12, ...   % Reduce tolerance for optimality
    'StepTolerance', 1e-12, ...         % Reduce step tolerance
    'Display', 'iter-detailed');       % Display detailed iteration information

% Optimization
[sigma_const, ~] = fmincon(fun, x0, [], [], [], [], lb, [], []);

MSE_const = fun(sigma_const);

%% Prices
[model_prices_const, model_volatilities_const] = calibration_const(sigma_const(1), sigma_const(2), F0, Strikes, TTM_call, discounts_call);

%% Plot
figure()
surf(Strikes, TTM_call, mkt_prices, 'FaceColor', 'blue', 'EdgeColor', 'none');
hold on;
surf(Strikes, TTM_call, model_prices_const, 'FaceColor', 'red', 'EdgeColor', 'none');

legend('Market prices', 'Model prices');
xlabel('Strikes');
ylabel('Time to Maturity (TTM)');
zlabel('Prices');
title('Comparison of Market and Model Prices');
hold off;

% Volatilities
figure()
surf(Strikes, TTM_call, mkt_vol, 'FaceColor', 'blue', 'EdgeColor', 'none'); % Market volatilities in blue
hold on
surf(Strikes, TTM_call, model_volatilities_const./sqrt(TTM_call), 'FaceColor', 'red', 'EdgeColor', 'none'); % Model volatilities in red

legend('Market implied volatility', 'Model volatility')
xlabel('Strike Prices')
ylabel('Time to Maturity (TTM)')
zlabel('Volatility')
title('Comparison of Market and Model Volatilities')
hold off

%% Point 4
fun = @(x) mean((mkt_prices - calibration_single_const(x(1:end-1), x(end), F0, Strikes, TTM_call, discounts_call)).^2, "all");

lb = zeros(length(TTM_call)+1,1);
x0 = max_vol*ones(length(TTM_call)+1,1);

% Optimization
[sigma_single_time, ~] = fmincon(fun, x0, [], [], [], [], lb, [], [], options);

MSE_single_time = fun(sigma_single_time);

%% Prices
[model_prices_single_time, model_volatilities_single_time] = calibration_single_const(sigma_single_time(1:end-1), sigma_single_time(end), F0, Strikes, TTM_call, discounts_call);

%% Plot
figure()
surf(Strikes, TTM_call, mkt_prices, 'FaceColor', 'blue', 'EdgeColor', 'none');
hold on;
surf(Strikes, TTM_call, model_prices_single_time, 'FaceColor', 'red', 'EdgeColor', 'none');

legend('Market prices', 'Model prices');
xlabel('Strikes');
ylabel('Time to Maturity (TTM)');
zlabel('Prices');
title('Comparison of Market and Model Prices');
hold off;

% Volatilities
figure()
surf(Strikes, TTM_call, mkt_vol, 'FaceColor', 'blue', 'EdgeColor', 'none'); % Market volatilities in blue
hold on
surf(Strikes, TTM_call, model_volatilities_single_time./sqrt(TTM_call), 'FaceColor', 'red', 'EdgeColor', 'none'); % Model volatilities in red

legend('Market implied volatility', 'Model volatility')
xlabel('Strike Prices')
ylabel('Time to Maturity (TTM)')
zlabel('Volatility')
title('Comparison of Market and Model Volatilities')
hold off

%% Point 5
fun = @(x) mean((mkt_prices - calibration_time_dep(x(:,1), x(:,2), F0, Strikes, TTM_call, discounts_call)).^2, "all");

lb = zeros(length(TTM_call),2);
x0 = max_vol*ones(length(TTM_call),2);

% Optimization
[sigma_double_time, ~] = fmincon(fun, x0, [], [], [], [], lb, [], [], options);

MSE_double_time = fun(sigma_double_time);

%% Prices
[model_prices_double_time, model_volatilities_double_time] = calibration_time_dep(sigma_double_time(:,1), sigma_double_time(:,2), F0, Strikes, TTM_call, discounts_call);

%% Plot
figure()
surf(Strikes, TTM_call, mkt_prices, 'FaceColor', 'blue', 'EdgeColor', 'none');
hold on;
surf(Strikes, TTM_call, model_prices_double_time, 'FaceColor', 'red', 'EdgeColor', 'none');

legend('Market prices', 'Model prices');
xlabel('Strikes');
ylabel('Time to Maturity (TTM)');
zlabel('Prices');
title('Comparison of Market and Model Prices');
hold off;

% Volatilities
figure()
surf(Strikes, TTM_call, mkt_vol, 'FaceColor', 'blue', 'EdgeColor', 'none'); % Market volatilities in blue
hold on
surf(Strikes, TTM_call, model_volatilities_double_time./sqrt(TTM_call), 'FaceColor', 'red', 'EdgeColor', 'none'); % Model volatilities in red

legend('Market implied volatility', 'Model volatility')
xlabel('Strike Prices')
ylabel('Time to Maturity (TTM)')
zlabel('Volatility')
title('Comparison of Market and Model Volatilities')
hold off

%% Final Plot
% Prices
figure()
surf(Strikes, TTM_call, mkt_prices, 'FaceColor', 'yellow', 'EdgeColor', 'none');
hold on;
surf(Strikes, TTM_call, model_prices_const, 'FaceColor', 'green', 'EdgeColor', 'none');
surf(Strikes, TTM_call, model_prices_single_time, 'FaceColor', 'red', 'EdgeColor', 'none');
surf(Strikes, TTM_call, model_prices_double_time, 'FaceColor', 'blue', 'EdgeColor', 'none');

xlabel('Strikes');
ylabel('Time to Maturity (TTM)');
zlabel('Prices');

legend('Mkt prices','Const volatility prices', 'Single time volatility prices', 'Double time volatility prices')

% Volatilities
figure()
surf(Strikes, TTM_call, mkt_vol, 'FaceColor', 'yellow', 'EdgeColor', 'none'); % Market volatilities in blue
hold on
surf(Strikes, TTM_call, model_volatilities_const./sqrt(TTM_call), 'FaceColor', 'green', 'EdgeColor', 'none');
surf(Strikes, TTM_call, model_volatilities_single_time./sqrt(TTM_call), 'FaceColor', 'red', 'EdgeColor', 'none');
surf(Strikes, TTM_call, model_volatilities_double_time./sqrt(TTM_call), 'FaceColor', 'blue', 'EdgeColor', 'none');

xlabel('Strikes');
ylabel('Time to Maturity (TTM)');
zlabel('Volatilities');

legend('Mkt volatility','Const volatility', 'Single time volatility', 'Double time volatility')

%% MC Down&In call pricing - constant

% Set the seed from now on
rng(10)

% Set the parameters
L = 450;
K = 500;
TTM_option = 0.5;

% Calculate the 6-month discount
zero_rate = interp1(discounts_frac,zero_rates,TTM_option);
discount = exp(-zero_rate.*TTM_option);

% Compute the difference between the reset tenors
tt = [TTM_call(1); TTM_call(2:end) - TTM_call(1:end-1)];

% Retrieve the calibrated parameters
sigma1 = sigma_const(1);
sigma2 = sigma_const(2);

% Compute the integral in case of constant volatility
sigma_hat = sqrt((sigma1^2 + sigma2^2).*tt);

% Simulation
Nsim = 1e6;
F_sim = simulateMC(F0, tt, sigma_hat, Nsim);

% Price the option
payoff = max(F_sim(:,end) - K, 0) .*(min(F_sim, [], 2)<=L);
priceMC_const = mean(discount * payoff)

%% MC Down&In call pricing - single time dependent

% Compute the integral by using previously calibrated volatilities
sigma_hat = sqrt(sigma_single_time(1:8)+sigma_single_time(end)^2*tt);

% Simulation
F_sim = simulateMC(F0, tt, sigma_hat, Nsim);

% Price the option
payoff = max(F_sim(:,end) - K, 0) .*(min(F_sim, [], 2)<=L);
priceMC_single = mean(discount * payoff)

%% MC Down&In call pricing - double time dependent

% Compute sigma_hat
sigma_hat = sqrt(sigma_double_time(1:8,1)+sigma_double_time(1:8,2));

% Simulation
F_sim = simulateMC(F0, tt, sigma_hat, Nsim);

% Price the option
payoff = max(F_sim(:,end) - K, 0) .*(min(F_sim, [], 2)<=L);
priceMC_double = mean(discount * payoff)

%% Constant vol the most reliable - different monitoring

% RMK. We have that in the time dependent case, the volatilities 
% have been calibrated each in tn-1, tn, thus I can't modify the reset
% dates. It is possible instead to modify the reset dates in the case I
% have constant volatity, as it does not depend on time. Thus I can refine
% my greed as much as I want. For instance, if I proceed with weekly
% monitoring.
% Because of this, we would trust this last one the most.

% Compute a weekly timespace
M = TTM_option*52; % change here if you want 100,500 ecc. timesteps
weeks = linspace(0, TTM_option , M+1);
tt = weeks(2:end) - weeks(1:end-1);

% Retrieve the calibrated parameters
sigma1 = sigma_const(1);
sigma2 = sigma_const(2);

% Compute the integral in case of constant volatility
sigma_hat = sqrt((sigma1^2 + sigma2^2).*tt);

% Simulation
F_sim = simulateMC(F0, tt, sigma_hat, Nsim);

% Price the option
payoff = max(F_sim(:,end) - K, 0).*(min(F_sim, [], 2)<=L);
priceMC_const_refined = mean(discount * payoff)

%% Exact formula

% Compute sigma (we do not use the ttm because it is inside the formula)
sigma_hat = sqrt(sigma1^2 + sigma2^2);

% EU call
price_EU = blkprice(F0, K, zero_rate, TTM_option, sigma_hat)

% Price the option
price_aux = blkprice(L^2/F0, K, zero_rate, TTM_option, sigma_hat);
price = (L/F0)^(2*(zero_rate - sigma_hat^2/2)/sigma_hat^2) * price_aux

% Price computed using market implied vol
sigma = mkt_vol(8,11);
price_aux = blkprice(L^2/F0, K, zero_rate, TTM_option, sigma);
price_table = (L/F0)^(2*(zero_rate - sigma^2/2)/sigma^2) * price_aux

toc