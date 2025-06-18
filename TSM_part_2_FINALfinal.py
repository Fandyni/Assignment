#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:12:55 2025

@author: Fandy
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.stats import skew, kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as sts
import statsmodels.api as sm
import numdifftools as nd
from scipy.optimize import minimize


def descriptive_data(returns): 
    '''
    Function to obtain the descriptive statistics of the original data y_t.
    '''
    print("Summary of data:")
    print("Mean:", round(np.mean(returns), 4))
    print("Min:", round(min(returns), 4))
    print("Max:", round(max(returns), 4))
    print("Variance:", round(np.var(returns), 4))
    print("Standard deviation:", round(np.std(returns), 4))
    print("Skewness:", round(skew(returns), 4))
    print("Kurtosis:", round(kurtosis(returns, fisher=True), 4))
    print("Ljung-Box p-value squared returns:", round(acorr_ljungbox(returns**2, lags=20).iloc[:,1][20], 4))
    jb_stat, jb_pval = sts.jarque_bera(returns)
    print("JB test, test statistic and p-value respectively:", round(jb_stat, 4), round(jb_pval, 4))
    
    return  


def plot_time_series(y, label, y_label):
    '''
    Function to plot the original time series y_t.
    '''
    plt.figure(figsize=(9, 3))
    plt.plot(y.index.values, y.values, label=label, linewidth=0.5, color="black")
    plt.xlabel("Time")
    plt.ylabel(y_label)
    plt.title(label+" over time")  
    plt.grid(True)
    plt.legend()
    plt.show()  
    
    return


def plot_x_t(x):
    '''
    Function to plot the transformed data x_t.
    '''
    plt.figure(figsize=(15, 3))
    plt.scatter(x.index, x, label='$x_t$', linewidth=0.5, color="black")
    plt.xlabel("Time")
    plt.ylabel("$x_t$")
    plt.title("$x_t$ over time")  
    plt.grid(True)
    plt.legend()
    plt.show() 
    
    return


def compute_x_t(y):
    '''
    Function to compute transformed data x_t. 
    '''
    return np.log((y - np.mean(y))**2)


def readfile(file):
    '''
    Function that reads the given data file and returns it. 
    '''

    df = pd.read_excel(file)    
    returns = df['GBPUSD']
    descriptive_data(returns)
    
    return returns


def part_c(x, kappa_ini, phi_ini, sigma2_eta_ini):
    '''
    Function for part c of the assignment.
    Computes the QML estimates. 
    '''
    initial_params = (kappa_ini, phi_ini, sigma2_eta_ini)
    
    bounds = [(None, None), (0, 0.99999), (0, None)]
   
    # minimize wrt params
    result = minimize(lambda params: -log_likelihood(x, params), initial_params, method='L-BFGS-B', bounds=bounds)  
    
    return result


def log_likelihood(y, params):
    '''
    Function to compute the log-likelihood and return it as a value (sum of log-likelihood values). 
    '''
    n = len(y)
    kappa, phi, sigma2_eta = (params)

    # run Kalman filter
    filtered_a, filtered_P, a, P, v, F, K = kalman_filter(y, kappa=kappa, phi=phi, sigma2_eta=sigma2_eta)
    
    llik = -(n / 2) * np.log(2 * np.pi) - (1 / 2) * np.sum(np.log(F[1:]) + (v[1:] ** 2) / F[1:])

    return llik


def log_likelihood_vector(y, params):
    '''
    Function to compute the log-likelihood and return it as a vector. This vector is needed for computing the standard errors.
    '''
    n = len(y)
    kappa, phi, sigma2_eta = (params)

    # run Kalman filter
    filtered_a, filtered_P, a, P, v, F, K = kalman_filter(y, kappa=kappa, phi=phi, sigma2_eta=sigma2_eta)  

    llik = -(n / 2) * np.log(2 * np.pi) - 0.5 * (np.log(F) + (v**2) / F)
    
    return llik


def kalman_filter(y, kappa, phi, sigma2_eta): 
    '''
    Function to run over the KF recursions for models with mean adjustments;
    The specific system matrices are explained in the corresponding comments. 
    Computes filtered_a, filtered_P, a, P, v, F and K.
    '''
    # initialize variables
    n = len(y)
    filtered_a, filtered_P = np.empty(n), np.empty(n)
    a, P = np.empty(n), np.empty(n) 
    v, F = np.empty(n), np.empty(n)
    K = np.empty(n)
   
    a[0] = 0
    P[0] = sigma2_eta / (1 - phi**2)
    
    for t in range(n-1): 
        # compute prediction error and Kalman gain prev. step
        v[t] = y[t] - a[t] - kappa # y_t - Z_t * a_t - d_t, where d_t = kappa
        F[t] = P[t] + H            # Z_t * P_t * Z_t' + H_t, where Z_t = 1
        K[t] = phi * (P[t] / F[t])

        # filtering step
        filtered_a[t] = a[t] + P[t] * (F[t])**-1 * v[t] # a_t + P_t * Z_t' * F_t^-1 * v_t, where Z_t = 1
        filtered_P[t] = P[t] - P[t] * (F[t])**-1 * P[t] # P_t - P_t * Z_t' * F_t^-1 * Z_t * P_t, where Z_t = 1

        # prediction step
        a[t + 1] = phi * filtered_a[t] # T_t * a_t|t + c_t, where T_t = phi and c_t = 0
        P[t + 1] = phi * filtered_P[t] * phi + sigma2_eta # T_t * P_t|t * T_t' + R_t * Q_t * R_t', where T_t = phi, R_t = 1 & Q_t = sigma_n

    v[n - 1] = y[n - 1] - a[n - 1] - kappa
    F[n - 1] = P[n - 1] + H
    
    filtered_a[n - 1] = a[n - 1] + P[n - 1] * (F[n - 1])**-1 * v[n - 1]
    filtered_P[n - 1] = P[n - 1] - P[n - 1] * (F[n - 1])**-1 * P[n - 1]
  
    return filtered_a, filtered_P, a, P, v, F, K 


def Kalman_Smoother(y, P, a, K, F, v, phi):
    '''
    Function to run over the KS recursions for models with mean adjustments;
    The specific system matrices are explained in the corresponding comments. 
    Computes a_hat, V, N and r.
    '''
    n = len(y)
    r = np.zeros(n)  
    N = np.zeros(n)  
    a_hat = np.zeros(n)  
    V = np.zeros(n)  
        
    r[-1] = 0
    N[-1] = 0
    
    for t in range(n - 1, -1, -1): 
        r[t-1] = F[t]**-1 * v[t] + (phi - K[t]) * r[t] # Z_t' * F_t^-1 * v_t + L_t' * r_t, where Z_t = 1 & L_t = phi - K_t
        N[t-1] = F[t]**-1 + (phi - K[t]) * N[t] * (phi - K[t]) # Z_t' * F_t^-1 * Z_t + L_t' * N_t * L_t, where Z_t = 1 & L_t = phi - K_t
 
        a_hat[t] = a[t] + P[t] * r[t-1]
        V[t] = P[t] - P[t] * N[t-1] * P[t]
        
    r_0 = F[0]**-1 * v[0] + (phi - K[0]) * r[0]  
    a_hat[0] = a[0] + P[0] * r_0
    
    N_0 = F[0]**-1 + (phi - K[0]) * N[0] * (phi - K[0])
    V[0] = P[0] - P[0] * N_0 * P[0]
        
    return a_hat, V, N, r


def plot_kalman(date, a, a_hat, x_t, kappa):
    '''
    Function to plot the filtered alpha_t, smoothed alpha_t, shifted smoothed alpha_t and transformed data x_t.
    '''
    fig, axs = plt.subplots(2, 1, figsize=(13, 8))   
    
    # Subplot 1: Contains both the filtered and smoothedestimates of αt
    axs[0].plot(date, a, label=r'filtered $\alpha_t$') 
    axs[0].plot(date, a_hat, label=r'Smoothed $\alpha_t$')
    axs[0].set_xlim(0, 945)
    axs[0].set_xticks(np.arange(0, 901, 100)) 
    axs[0].set_title('Kalman filter and smoother')
    axs[0].set_ylabel('Values')
    axs[0].legend()   

    # Subplot 2: Contains the transformed data xt, together with the shifted smoothing αt, αt+k
    axs[1].scatter(date, x_t, label=r'Data $x_t$')
    axs[1].plot(date, a_hat+kappa, color = "black", label=r'Shifted smoothed $\alpha_t$')
    axs[1].set_xlim(0, 945)
    axs[1].set_xticks(np.arange(0, 901, 100)) 
    axs[1].set_title(r'Transformed data and shifted smoothed $\alpha_t$')
    axs[1].set_ylabel('Values')
    axs[1].legend()

    plt.tight_layout()   
    plt.show()
    return


def simulate_data(n, kappa, phi, sigma_n):
    '''
    Function to simulate data from the model for x_t --> used for validation of QML implementation.
    '''
    np.random.seed(1)
    
    alpha = np.zeros(n)
    x = np.zeros(n-1)
    
    eta = np.random.normal(0, np.sqrt(sigma_n), n)
    xi = np.random.normal(0, np.sqrt((np.pi**2) / 2), n)
    
    for t in range(n-1):
        x[t] = kappa + alpha[t] + xi[t]
        alpha[t + 1] = phi * alpha[t] + eta[t]
        
    return x


def evaluate_z_A(g, yt, mu, sigma):
    '''
    Function that computes z and A as explained in the lectures. 
    '''
    # Compute standardized observations
    yt_standardized = (yt - mu) /sigma  # Standardize observations
    
    # Compute A_t and z_t based on equations in the slides
    A = 2 * np.exp(g) / (yt_standardized ** 2)  
    z = g + 1 - np.exp(g) / (yt_standardized ** 2)

    return z, A


def mode_estimation(y, kappa, phi, sigma2_eta):
    '''
    Function that addopts the mode estimation algorithm
    Computes the estimated mode and the history of guesses g. 
    '''
    n = len(y)
    sigma = np.exp((kappa + 1.27) / 2)

    g = np.log(y**2)
    g_plus = np.zeros(n)
    g_history = []  # List to store mode estimates from each iteration

    conv_bound = 10**-7
    y_barbar = (y - np.mean(y)) / sigma

    while True:
        A = 2 * np.exp(g) / (y_barbar) ** 2
        z = g + 1 - np.exp(g) / (y_barbar) ** 2

        sigma2_epsilon = A

        # Initialize Kalman filter variables
        a = np.zeros(n)  # Filtered state estimate
        P = np.zeros(n)  # Filtered state variance
        v = np.zeros(n)  # Prediction errors
        F = np.zeros(n)  # Prediction error variance
        K = np.zeros(n)  # Kalman gain

        # Initial values
        a[0] = 0
        P[0] = sigma2_eta / (1 - phi**2)

        # Kalman filter recursions
        for t in range(n-1):
            
            v[t] = z[t] - a[t]
            F[t] = P[t] + sigma2_epsilon[t]
            K[t] = P[t] / F[t]
            a_t_t = a[t] + K[t] * v[t]
            P_t_t = K[t] * sigma2_epsilon[t]
            a[t + 1] = a_t_t
            P[t + 1] = P_t_t + sigma2_eta

        v[n - 1] = z[n - 1] - a[n - 1]
        F[n - 1] = P[n - 1] + sigma2_epsilon[n - 1]
        K[n - 1] = P[n - 1] / F[n - 1]

        # Kalman smoother recursions
        alpha_hat = np.zeros(n)
        V = np.zeros(n)
        r = np.zeros(n + 1)
        N = np.zeros(n + 1)

        r[-1] = 0
        N[-1] = 0

        for t in range(n, 0, -1):
            
            r[t - 1] = v[t - 1] / F[t - 1] + (1 - K[t - 1]) * r[t]
            N[t - 1] = 1 / F[t - 1] + (1 - K[t - 1]) ** 2 * N[t]
            alpha_hat[t - 1] = a[t - 1] + P[t - 1] * r[t - 1]
            V[t - 1] = P[t - 1] - P[t - 1] ** 2 * N[t - 1]

        g_plus = alpha_hat

        # Store the mode (g_plus) at each iteration
        g_history.append(g_plus)

        if np.all(np.abs(g_plus - g) <= conv_bound):
            mode = g_plus
            break
        else:
            g = g_plus

    return mode, g_history


def plot_mode_and_smoothed_state(mode, smoothed_states):
    '''
    Function to plot the estimated mode and compare it with the smoothed state estimates obtained in part c.
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(mode)), mode, color="gray", linewidth=0.6, label="Mode Estimates")
    plt.plot(range(len(smoothed_states)), smoothed_states, color="blue", linewidth=0.6, label="Smoothed QML Estimates")
    plt.xlabel("Time")
    plt.ylabel("Estimates")
    plt.legend()
    plt.savefig("mode_estimates_plot.png")
    plt.show()
 
    return


# Define the Bootstrap Particle Filter
def bootstrap_particle_filter(y, kappa, phi, sigma_eta):
    '''
    Function for the bootstrap filter
    Computes the filtered alpha's and alpha particles.
    '''
    np.random.seed(1)
    
    M = 10000
    sigma = np.exp((kappa + 1.27) / 2)
    n = len(y) # Number of observations
    alpha_particles = np.zeros((M, n)) # Particles for alpha_t
    weights = np.zeros((M, n)) # Weights
    
    # Initialize particles (assume normal prior N(0, sigma_eta^2))
    alpha_particles[:, 0] = np.random.normal(0, np.sqrt(sigma_eta**2 / (1 - phi**2)), M)
    
    # Iterate over time
    for t in range(1, n):
        # Step 1: Propagate particles (State transition)
        alpha_particles[:, t] = np.random.normal(phi * alpha_particles[:, t - 1], np.sqrt(sigma_eta), M)
        
        # Step 2: Compute importance weights using likelihood p(y_t | alpha_t)
        likelihoods = np.exp(-0.5 * np.log(2*np.pi*(sigma**2)) -0.5 * alpha_particles[:, t] - (1 / (2 * (sigma**2))) * \
                             np.exp(-alpha_particles[:, t]) * (y[t] - np.mean(y))**2)
        
        weights[:, t] = likelihoods / np.sum(likelihoods) # Normalize weights
        
        # Step 3: Resample particles
        resample_indices = np.random.choice(M, M, p=weights[:, t])
        alpha_particles[:, t] = alpha_particles[resample_indices, t]
        weights[:, t] = 1.0 / M  # Reset weights to uniform after resampling
    
    # Compute filtered estimate (weighted mean of particles)
    alpha_filtered = np.sum(alpha_particles * weights, axis=0)
    
    return alpha_filtered, alpha_particles


def main():
    
    global H
    file = 'sv_log_returns.xlsx'
    
    # part a
    returns = readfile(file)
    
    # part b  
    plot_time_series(returns, "Log returns", "Log return")
    x_t = compute_x_t(returns)
    plot_x_t(x_t)
   
    # part c
    H = 4.93
    kappa_ini = np.mean(x_t)
    autocov = sm.tsa.acovf(x_t, fft=False)
    phi_ini = autocov[4]/autocov[3] ### increase lag until phi is between 0 and 1    
    sigma2_eta_ini = (1 - phi_ini**2) * (np.var(x_t, ddof=1) - ((np.pi**2)/2))
    
    result = part_c(x_t, kappa_ini, phi_ini, sigma2_eta_ini)
    
    # Obtaining standard errors of estimated parameters
    vLogL_func = lambda p: log_likelihood_vector(x_t, p)
    G = nd.Jacobian(vLogL_func)(result.x)
    G = np.squeeze(G)
    G = np.array(G)
    outer_product = np.linalg.inv(np.dot(G.T, G))
    SEs = np.sqrt(np.diag(outer_product))
        
    print("\n------------- QML Estimation results -------------")
    log_likelihood_value = -result.fun
    print("\nLog-Likelihood value:", np.round(log_likelihood_value, 4))
    
    print("Estimates (kappa, phi, sigma_eta):", 
          np.round(result.x[0], 4), 
          np.round(result.x[1], 4), 
          np.round(result.x[2], 4))
    print("Standard Errors:", np.round(SEs, 4))

    sigma2 = np.exp(result.x[0]) + np.exp(1.27) 
    print("sigma original SV model:", np.round(np.sqrt(sigma2), 4))
    
    print("\n------------- Check of the implementation -------------\n")
    
    print("MoM estimate kappa:", np.round(kappa_ini, 4), "versus QML estimate:", np.round(result.x[0], 4))
    
    sigma_eta_MoM = (1 - (result.x[1]**2)) * (np.var(x_t, ddof=1) - ((np.pi**2) / 2))
    
    print("\nMoM estimate of sigma_eta:", np.round(sigma_eta_MoM, 4), "versus QML estimate:", np.round(result.x[2], 4))
    print("Fraction:", np.round((result.x[2] / sigma_eta_MoM), 4))
    
    n = 10000
    x_simulated = simulate_data(n, kappa_ini, phi_ini, sigma2_eta_ini)
    simulated_result = part_c(x_simulated, kappa_ini, phi_ini, sigma2_eta_ini)
    
    # Obtaining standard errors of estimated parameters
    vLogL_func = lambda p_sim: log_likelihood_vector(x_simulated, p_sim)
    G_sim = nd.Jacobian(vLogL_func)(simulated_result.x)
    G_sim = np.squeeze(G_sim)
    G_sim = np.array(G_sim)
    outer_product_sim = np.linalg.inv(np.dot(G_sim.T, G_sim))
    SEs_sim = np.sqrt(np.diag(outer_product_sim))
    
    np.set_printoptions(precision=4)

    print("Estimated Parameters using simulation:", np.round(simulated_result.x[0], 4), np.round(simulated_result.x[1], 4),
          np.round(simulated_result.x[2], 4))
    print("Standard errors using simulation: [", np.round(SEs_sim[0], 4), np.round(SEs_sim[1], 4), np.round(SEs_sim[2], 4), "]")
        
    # part d    
    filtered_a, filtered_P, a, P, v, F, K = kalman_filter(x_t, result.x[0], result.x[1], result.x[2])
    a_hat, V, N, r = Kalman_Smoother(x_t, P, a, K, F, v, result.x[1])
    
    x = np.arange(0, len(x_t))
    plot_kalman(x, filtered_a, a_hat, x_t, result.x[0])
     
    # part e
    mode, g_history = mode_estimation(returns, result.x[0], result.x[1], result.x[2])  
    plot_mode_and_smoothed_state(mode, a_hat)
 
    # part f
    alpha_filtered, _ = bootstrap_particle_filter(returns, result.x[0],result.x[1], result.x[2])
    
    # Plot the results
    plt.figure(figsize=(20, 8)) 
    plt.plot(filtered_a, label='Filtered alpha_t (part c)', linewidth=2, color='blue')
    plt.plot(alpha_filtered, label='Filtered alpha_t (Bootstrap Filter)', linewidth=2, color='red', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Alpha')
    plt.legend()
    plt.title('Bootstrap Particle Filter for Stochastic Volatility Model')
    plt.show()
    
    
if __name__ == "__main__":
    main()
