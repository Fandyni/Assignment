#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:32:38 2025

@author: aimee
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from statsmodels.tsa.stattools import acovf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

def plot_filter(date, a, y, CI_U, CI_L, P, v, F):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))   
    fig.suptitle('Kalman Filter', fontsize=16)
    axs[0, 0].plot(date, a, label=r'$a_t$', color='blue')  
    axs[0, 0].plot(date, CI_U, label=r'95% CI_U', color='black', alpha=0.5)  
    axs[0, 0].plot(date, CI_L, label=r'95% CI_L', color='black', alpha=0.5)  
    axs[0, 0].scatter(date, y, label=r'$y_t$', color='red', marker='o')
    axs[0, 0].set_title('Nile data and output of Kalman filter')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].legend()   

    # Subplot 2: P_t
    axs[0, 1].plot(date, P, color='orange', label=r'$P_t$')
    axs[0, 1].set_title(r'Filtered state variance $P_t$')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Values')
    axs[0, 1].legend()

    # Subplot 3: v_t
    axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[1, 0].plot(date, v, color='green', label=r'$v_t$')
    axs[1, 0].set_title(r'Prediction errors $v_t$')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Values')
    axs[1, 0].legend()

    # Subplot 4: F_t
    axs[1, 1].plot(date, F, color='red', label=r'$F_t$')
    axs[1, 1].set_title(r'Prediction variance $F_t$')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Values')
    axs[1, 1].legend()

    plt.tight_layout()   
    plt.show()
    return


def plot_smoother(date, r, a_hat, CI_U, CI_L, N, V, y):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))   
    fig.suptitle('Kalman Smoother', fontsize=16)
    axs[0, 0].plot(date, a_hat, label=r'$\hat{\alpha}_t$', color='blue')  
    axs[0, 0].plot(date, CI_U, label=r'95% CI_U', color='black', alpha=0.5)  
    axs[0, 0].plot(date, CI_L, label=r'95% CI_L', color='black', alpha=0.5)  
    axs[0, 0].scatter(date, y, label=r'$y_t$', color='red', marker='o')
    axs[0, 0].set_title('Nile data and output of Kalman smoother')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].legend()   

    # Subplot 2: V_t
    axs[0, 1].plot(date, V, color='orange', label=r'$V_t$')
    axs[0, 1].set_title(r'Smoothed state variance $V_t$')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Values')
    axs[0, 1].legend()

    # Subplot 3: r_t
    axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[1, 0].plot(date[:-1] , r[:-1] , color='green', label=r'$r_t$')
    axs[1, 0].set_title(r'Smoothing cumulant $r_t$')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Values')
    axs[1, 0].legend()

    # Subplot 4: N_t
    axs[1, 1].plot(date[:-1] , N[:-1] , color='red', label=r'$N_t$')
    axs[1, 1].set_title(r'Smoothing variance cumulant $N_t$')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Values')
    axs[1, 1].legend()

    plt.tight_layout()   
    plt.show()
    return


def plot_missing(a, y, P, a_hat, V):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))   
    fig.suptitle('Kalman Smoother and filter with missing variables', fontsize=16)
    
    axs[0, 0].plot(date[1:], a[1:], label=r'$a_t$', color='blue')  
    axs[0, 0].plot(date[1:], y[1:], label=r'$y_t$', color='red', marker='o')
    axs[0, 0].set_title('Nile data and output of Kalman filter')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].legend()   

    # Subplot 2: P_t
    axs[0, 1].plot(date[1:], P[1:], color='orange', label=r'$P_t$')
    axs[0, 1].set_title(r'Filtered state variance $P_t$')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Values')
    axs[0, 1].legend()
    
    # Subplot 3:a hat
    axs[1, 0].plot(date, a_hat, label=r'$\hat{\alpha}_t$', color='blue')  
    axs[1, 0].plot(date, y, label=r'$y_t$', color='red', marker='o')
    axs[1, 0].set_title('Nile data and output of Kalman smoother')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Values')
    axs[1, 0].legend()   

    # Subplot 4: V_t
    axs[1, 1].plot(date, V, color='orange', label=r'$V_t$')
    axs[1, 1].set_title(r'Smoothed state variance $V_t$')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Values')
    axs[1, 1].legend() 

    plt.tight_layout()   
    plt.show()
    
    return


def plot_sts_forecast_errors(F, v):
    
    e = v / np.sqrt(F)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))   
    fig.suptitle('Diagnostic plots for standardised prediction errors', fontsize=16)
    
    # Subplot 1: e_t
    axs[0, 0].plot(date, e, label=r'$e_t$', color='purple')  
    axs[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[0, 0].set_title('Standardised residuals')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].legend()   

    # Subplot 2: Histogram with estimated density
    axs[0, 1].hist(e, bins=13, density=True, alpha=0.6, color='white', edgecolor='black')
    kde = gaussian_kde(e)
    x_vals = np.linspace(min(e), max(e), 10000)
    axs[0, 1].plot(x_vals, kde(x_vals), color='black')  # Estimated density
    axs[0, 1].set_title('Histogram and Estimated Density')
            
    # Subplot 3: QQ plot
    ordered_residuals = np.sort(e)
    n = len(e)
    theoretical_quantiles = norm.ppf((np.arange(1, n + 1) - 0.5) / n) 
    axs[1, 0].plot(theoretical_quantiles, ordered_residuals, color='green', label='Sorted residuals')
    axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    q_min, q_max = np.min(theoretical_quantiles), np.max(theoretical_quantiles)
    axs[1 ,0].plot([q_min, q_max], [q_min, q_max], 'r-', label='45-degree line')
    axs[1, 0].set_title("QQ Plot of ordered residuals")
    axs[1, 0].legend()

    # Subplot 4: Correlogram
    lags = 12
    acf_values = acf(e, nlags=lags)
    
    axs[1, 1].bar(range(1, lags + 1), acf_values[1:], color='blue', alpha=0.7)
    axs[1, 1].axhline(y=0, color='black', linewidth=1)
    axs[1, 1].set_title('Correlogram')
    axs[1, 1].set_xlabel('Lag')
    axs[1, 1].set_ylabel('Autocorrelation')
    axs[1, 1].set_yticks([0.5, 0.0, -0.5])

    plt.tight_layout()   
    plt.show()
    
    return


def plot_sts_smoothed_residuals(F, v, K, N, r):
    
    u_t = F**(-1) * v - K * r
    D_t = F**(-1) + K**2 * N 
    
    u_t_star = D_t**(-1/2) * u_t
    r_t_star = np.var(r)**(-1/2) * r
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))   
    fig.suptitle('Diagnostic plots for standardised prediction errors', fontsize=16)
    
    # Subplot 1: u_t_star
    axs[0, 0].plot(date, u_t_star, label=r'$u_t^{*}$', color='green')  
    axs[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[0, 0].set_title('Smoothed observation residuals $u_{t}^{*}$')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].legend()   

    # Subplot 2: Histogram with estimated density u_t_star
    axs[0, 1].hist(u_t_star, bins=13, density=True, alpha=0.6, color='white', edgecolor='black')
    kde = gaussian_kde(u_t_star)
    x_vals = np.linspace(min(u_t_star), max(u_t_star), 10000)
    axs[0, 1].plot(x_vals, kde(x_vals), color='black')  # Estimated density
    axs[0, 1].set_title('Histogram and Estimated Density $u_t^{*}$')
        
    # Subplot 3: r_t_star
    axs[1, 0].plot(date[:-1], r_t_star[:-1], label=r'$r_t^{*}$', color='purple')  
    axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[1, 0].set_title('Smoothed state residuals $r_{t}^{*}$')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Values')
    axs[1, 0].legend()   

    # Subplot 4: Histogram with estimated density r_t_star
    axs[1, 1].hist(r_t_star, bins=13, density=True, alpha=0.6, color='white', edgecolor='black')
    kde = gaussian_kde(r_t_star)
    x_vals = np.linspace(min(r_t_star), max(r_t_star), 10000)
    axs[1, 1].plot(x_vals, kde(x_vals), color='black')  # Estimated density
    axs[1, 1].set_title('Histogram and Estimated Density $r_t^{*}$')

    plt.tight_layout()   
    plt.show()

    return


def plot_smooth_disturbence(e_hat, Var_e, n_hat, Var_n):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))   
    fig.suptitle('Smooth disturbances', fontsize=16)
    
    # Subplot 1: obs error e_hat
    axs[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[0, 0].plot(date, e_hat, label=r'$\hat{\epsilon}_t$', color='blue')  
    axs[0, 0].set_title('Observation error $\hat{\epsilon}_t$')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].legend()   

    # Subplot 2: observation error variance var(e_hat| Y_n)
    axs[0, 1].plot(date, Var_e, color='orange', label=r'VAR($\epsilon_t| Y_n$)')
    axs[0, 1].set_title(r'Observation error variance VAR($\epsilon_t| Y_n$)')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Values')
    axs[0, 1].legend()

    # Subplot 3: state error n_t
    axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[1, 0].plot(date, n_hat, color='green', label=r'$\hat{\eta}_t$')
    axs[1, 0].set_title(r'State error $\hat{\eta}_t$')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Values')
    axs[1, 0].legend()

    # Subplot 4:state error variance Var(ηt|Yn)
    axs[1, 1].plot(date, Var_n, color='red', label=r'VAR($\eta_t| Y_n$)')
    axs[1, 1].set_title(r'State error variance VAR($\eta_t| Y_n$)')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Values')
    axs[1, 1].legend()

    plt.tight_layout()   
    plt.show()
    
    return


def Smooth_disturbances(y, P, a, K, F, v, N, r):
    
    e_hat = np.zeros(len(y))
    Var_e = np.zeros(len(y))
    n_hat = np.zeros(len(y))
    Var_n = np.zeros(len(y))
    
    for t in range(len(date) - 1, -1, -1):  
        e_hat[t] = sigma2_e * ((F[t]**-1) * v[t] - K[t] * r[t])
        Var_e[t] = sigma2_e - (sigma2_e**2) * ((F[t]**-1) + (K[t]**2) * N[t])
        n_hat[t] = sigma2_n * r[t]
        Var_n[t] = sigma2_n - (sigma2_n**2) * N[t]
   
    plot_smooth_disturbence(e_hat, np.sqrt(Var_e), n_hat, np.sqrt(Var_n))
    
    return


def Kalman_Filter(y, plot): 
    
    a1 = 0
    P1=10**7
     
    v = np.zeros(len(date))
    P = np.zeros(len(date))
    F = np.zeros(len(date))
    K = np.zeros(len(date))
    a = np.zeros(len(date))
    
    a[0] = a1
    P[0] = P1 
    
    for t in range(len(date)):
        if np.isnan(y[t]): 
            F[t] = P[t] + sigma2_e
            K[t] = P[t] / F[t] 
        
            if t+1 < len(date) and np.isnan(y[t]):   
                a[t+1] = a[t] 
                P[t+1] = P[t] + sigma2_n       
        else:
            v[t] = y[t] - a[t]
            F[t] = P[t] + sigma2_e
            K[t] = P[t] / F[t] 
            
            if t+1 < len(date):
                a[t+1] = a[t] + K[t] * v[t]
                P[t+1] = K[t] * sigma2_e + sigma2_n
            

    if plot == 'yes':
        CI_U =  a + 1.645 * np.sqrt(P)
        CI_L =  a - 1.645 * np.sqrt(P)
        plot_filter(date[1:], a[1:], y[1:],CI_U[1:],CI_L[1:], P[1:], v[1:], F[1:])
        
    return P, a, K, F, v


def Kalman_Smoother(y, P, a, K, F, v, plot):
    r = np.zeros(T)  
    N = np.zeros(T)  
    a_hat = np.zeros(T)  
    V = np.zeros(T)  
        
    r[-1] = 0
    N[-1] = 0
    
    for t in range(len(date) - 1, -1, -1):  
        if np.isnan(y[t]): 
            r[t-1] =  r[t]  
            N[t-1] =  N[t]
        else:
            r[t-1] =  F[t]**-1 * v[t] + (1 - K[t]) * r[t]  
            N[t-1] =  F[t]**-1 + ((1 - K[t]) **2) * N[t]
            
        a_hat[t] = a[t] + P[t] * r[t-1]
        V[t] = P[t] - (P[t] ** 2) * N[t-1]
        
    r_0 = F[0]**-1 * v[0] + (1 - K[0]) * r[0]  
    a_hat[0] = a[0] + P[0] * r_0
    
    N_0 = F[0]**-1 + ((1 - K[0]) **2) * N[0]
    V[0] = P[0] - (P[0] ** 2) * N_0
    
    if plot == 'yes':
        CI_U =  a_hat + 1.645 * np.sqrt(V)
        CI_L =  a_hat - 1.645 * np.sqrt(V)
        plot_smoother(date, r, a_hat, CI_U, CI_L, N, V, y)  
        
    return a_hat, V, N, r


def Missing_data(data):
    y = np.array(data, dtype=float) 
    y[20:40] = np.nan
    y[60:80] = np.nan
    
    P, a, K, F, v = Kalman_Filter(y, 'no')
    a_hat, V, _, _ = Kalman_Smoother(y, P, a, K, F, v, 'no')
    
    plot_missing(a, y, P, a_hat, V)
    return


def plot_forecast(years, a, y, P, F):
    CI_U = y + 0.5 * np.sqrt(P)
    CI_L = y - 0.5 * np.sqrt(P)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))   
    fig.suptitle('Forecast', fontsize=16)
    axs[0, 0].scatter(years[:100], a[:100], label=r'$a_t$', color='blue', s=10)  
    axs[0, 0].plot(years[100:], CI_U[100:], label=r'50% CI_U', color='black', alpha=0.5, linewidth=0.5)  
    axs[0, 0].plot(years[100:], CI_L[100:], label=r'50% CI_L', color='black', alpha=0.5, linewidth=0.5)  
    axs[0, 0].plot(years, y, label=r'$y_t$', color='red', linewidth=0.8 )
    axs[0, 0].set_title('Nile data and output of Kalman filter')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].legend()   

    # Subplot 2: P_t
    axs[0, 1].plot(years, P, color='orange', label=r'$P_t$')
    axs[0, 1].set_title(r'Filtered state variance $P_t$')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Values')
    axs[0, 1].legend()

    # Subplot 3: forecast 
    axs[1, 0].plot(years, y, color='green', label=r'$v_t$')
    axs[1, 0].set_title(r'Prediction errors $y_t$')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Values')
    axs[1, 0].legend() 

    # Subplot 4: F_t
    axs[1, 1].plot(years, F, color='red', label=r'$F_t$')
    axs[1, 1].set_title(r'Prediction variance $F_t$')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Values')
    axs[1, 1].legend()

    plt.tight_layout()   
    plt.show()
    
    return


def plot_simulations(alpha_hat, alpha_plus, alpha_tilde, epsilon_hat, epsilon_tilde, eta_hat, eta_tilde):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))   
    fig.suptitle('Simulation', fontsize=16)
    
    # Subplot 1: alpha_hat and alpha_plus
    axs[0, 0].plot(date, alpha_hat, label=r'$\hat{\alpha}_t$', color='blue')  
    axs[0, 0].scatter(date, alpha_plus, label=r'$\alpha_{t}^{+}$', color='orange', s=10)  
    axs[0, 0].set_title(r'Smoothed state and sample $\alpha_{t}^{+}$')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].legend()   

    # Subplot 2: alpha_hat and alpha_tilde
    axs[0, 1].plot(date, alpha_hat, label=r'$\hat{\alpha}_t$', color='blue')
    axs[0, 1].scatter(date, alpha_tilde, label=r'$\tilde{\alpha}_{t}$', color='orange', s=10)
    axs[0, 0].set_title(r'Smoothed state and sample $\tilde{\alpha}_{t}$')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Values')
    axs[0, 1].legend()

    # Subplot 3: epsilon_hat and epsilon_tilde 
    axs[1, 0].plot(date, epsilon_hat, color='green', label=r'$\hat{\epsilon}_{t}$')
    axs[1, 0].scatter(date, epsilon_tilde, label=r'$\tilde{\epsilon}_{t}$', color='red', s=10)
    axs[1, 0].set_title(r'Smoothed observation error and sample $\tilde{\epsilon}_{t}$')
    axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Values')
    axs[1, 0].legend() 

    # Subplot 4: eta_hat and eta_tilde 
    axs[1, 1].plot(date, eta_hat, color='green', label=r'$\hat{\eta}_{t}$')
    axs[1, 1].scatter(date, eta_tilde, label=r'$\tilde{\eta}_{t}$', color='red', s=10)
    axs[1, 1].set_title(r'Smoothed state error and sample $\tilde{\eta}_{t}$')
    axs[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Values')
    axs[1, 1].legend()

    plt.tight_layout()   
    plt.show()
    
    return


def simulate(y, P, a_hat, K, F, v, r):
    np.random.seed(500)
    n = len(y)
    
    u_t = F**(-1) * v - K * r
    epsilon_hat = sigma2_e * u_t
    
    eta_hat = sigma2_n * r
    
    epsilon_plus = np.random.normal(loc=0, scale=np.sqrt(sigma2_e), size=n)
    eta_plus = np.random.normal(loc=0, scale=np.sqrt(sigma2_n), size=n)
    
    y_plus = np.zeros(n)
    alpha_plus = np.zeros(n)
    alpha_plus[0] = y[0]
    
    for t in range(n):
        y_plus[t] = alpha_plus[t] + epsilon_plus[t]
        
        if t+1 < n:
            alpha_plus[t+1] = alpha_plus[t] + eta_plus[t]

    P_plus, a_plus, K_plus, F_plus, v_plus = Kalman_Filter(y_plus, 'no')    
    a_hat_plus, V_plus, N_plus, r_plus = Kalman_Smoother(y_plus, P_plus, a_plus, K_plus, F_plus, v_plus, 'no')
        
    u_t_plus = F_plus**(-1) * v_plus - K_plus * r_plus
    epsilon_hat_plus = sigma2_e * u_t_plus
    epsilon_tilde = epsilon_plus - epsilon_hat_plus + epsilon_hat
    
    alpha_tilde = y - epsilon_tilde    
    eta_tilde = np.zeros(n)
    
    for j in range(n):
        if j+1 < n:
            eta_tilde[j] = alpha_tilde[j+1] - alpha_tilde[j]
    
    plot_simulations(a_hat, alpha_plus, alpha_tilde, epsilon_hat, epsilon_tilde, eta_hat, eta_tilde)
    
    return


def Forecasting(data):
   
    years = np.arange(1871, 1971)  
    new_years = np.arange(1971, 2001)   
    years = np.concatenate((years, new_years))   
    
    y = np.array(data, dtype=float) 
    nans = np.full(30, np.nan)
    
    y = np.concatenate((y,nans))
    
    P, a, K, F, v = Kalman_Filter(y, 'no')
    
    a1 = 0
    P1=10**7
     
    v = np.zeros(len(y))
    P = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))
    a = np.zeros(len(y))
    
    a[0] = a1
    P[0] = P1 
    
    for t in range(len(y)):
        if np.isnan(y[t]): 
            F[t] = P[t] + sigma2_e
            K[t] = P[t] / F[t] 
        
            if t+1 < len(y) and np.isnan(y[t]):   
                a[t+1] = a[t] 
                P[t+1] = P[t] +  sigma2_n       
        else:
            v[t] = y[t] - a[t]
            F[t] = P[t] + sigma2_e
            K[t] = P[t] / F[t] 
            
            if t+1 < len(y):
                a[t+1] = a[t] + K[t] * v[t]
                P[t+1] = K[t] * sigma2_e + sigma2_n
    
    y[100:130] =  a[100:130]   
    plot_forecast(years[1:], y[1:], a[1:], P[1:], F[1:])  
    return


def function1(y):
    options = {'disp':True,
               'maxiter':200}

    # initialize parameters using method of moments estimates
    sigma2_eps_start = -acovf(np.diff(y), nlag=1)[1]
    sigma2_eta_start = np.var(np.diff(y)) + 2 * acovf(np.diff(y), nlag=1)[1]

    # minimize w.r.t log(params), this gives better results in practice
    result = minimize(lambda params: -log_likelihood(y, params), (np.log(sigma2_eps_start), np.log(sigma2_eta_start)), options=options, method='BFGS')
    
    
    return result

def log_likelihood(y, log_params):
    # initialize variables
    n = len(y)
    sigma2_eps, sigma2_eta = np.exp(log_params)

    # run Kalman filter
    filtered_a, filtered_P, a, P, v, F, K = kalman_filter(y, a_1=0, P_1=10**7, sigma2_eps=sigma2_eps, sigma2_eta=sigma2_eta)

    llik = -(n / 2) * np.log(2 * np.pi) - (1 / 2) * np.sum(np.log(F[1:]) + v[1:] ** 2 / F[1:])
    return llik


def kalman_filter(y, a_1, P_1, sigma2_eps, sigma2_eta):  # works ONLY for LLM's (general version not implemented yet)
    # initialize variables
    n = len(y)
    filtered_a, filtered_P = np.empty(n), np.empty(n)
    a, P = np.empty(n), np.empty(n)  # 1-step ahead predictions
    v, F = np.empty(n), np.empty(n)
    K = np.empty(n)

    a[0], P[0] = a_1, P_1

    for t in np.arange(n - 1):
        if not np.isnan(y[t]):
            # compute prediction error and Kalman gain prev. step
            v[t] = y[t] - a[t]
            F[t] = P[t] + sigma2_eps
            K[t] = P[t] / F[t]

            # filtering step
            filtered_a[t] = a[t] + K[t] * v[t]
            filtered_P[t] = K[t] * sigma2_eps

            # prediction step
            a[t + 1] = filtered_a[t]
            P[t + 1] = filtered_P[t] + sigma2_eta
        else:
            # handle v, F and K taking into account missing obs.
            v[t] = np.nan
            F[t] = np.inf
            K[t] = 0

            # filtering step
            filtered_a[t] = a[t]
            filtered_P[t] = P[t]

            # prediction step
            a[t + 1] = a[t]
            P[t + 1] = P[t] + sigma2_eta

    v[n - 1] = y[n - 1] - a[n - 1]
    F[n - 1] = P[n - 1] + sigma2_eps
    K[n - 1] = P[n - 1] / F[n - 1]

    filtered_a[n - 1] = a[n - 1] + K[n - 1] * v[n - 1]
    filtered_P[n - 1] = K[n - 1] * sigma2_eps

    return filtered_a, filtered_P, a, P, v, F, K



def main():
    
    global T, date
    data_file = pd.read_excel("Nile.xlsx")
    
    T = len(data_file)
    
    date = data_file.iloc[:,0].values 
    y = data_file.iloc[:,1].values 
    
    print("testtt", y)
    
    
    ##MLE part
    
    est_result = function1(y)
    sigma2_eps, sigma2_eta = np.exp(est_result.x)
    
    
    

    print(f'message: {est_result.message} \n llik: {-est_result.fun} \n sigma2_eps: {sigma2_eps} \n '
          f'sigma2_eta: {sigma2_eta} \n iterations: {est_result.nit} \n evaluations: {est_result.nfev} \n '
          f'success: {est_result.success}')
    
    
    ##part plots
    global sigma2_e, sigma2_n
  
    sigma2_e = 15099
    sigma2_n = 1469.1
    
    P, a, K, F, v = Kalman_Filter(y, 'yes')  
    a_smoother, V_smoother, N, r = Kalman_Smoother(y, P, a, K, F, v, 'yes')
    Smooth_disturbances(y, P, a, K, F, v, N, r)
    simulate(y, P, a_smoother, K, F, v, r)
    
    Missing_data(y)
    Forecasting(y)
    
    plot_sts_forecast_errors(F, v)
    plot_sts_smoothed_residuals(F, v, K, N, r)
    
if __name__ == "__main__":
    main()