import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import shap
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization

# Define stock tickers for Indian market
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']
returns = data.pct_change().dropna()

# Define portfolio optimization function
def portfolio_performance(weights, returns):
    weights = np.array(weights)
    port_return = np.sum(returns.mean() * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = port_return / port_volatility
    return sharpe_ratio

# Optimize portfolio using Bayesian Optimization
def optimize_portfolio():
    def objective(**weights):
        weight_array = np.array(list(weights.values()))
        return portfolio_performance(weight_array, returns)
    
    pbounds = {f'w{i}': (0, 1) for i in range(len(tickers))}
    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=50)
    
    best_weights = np.array([optimizer.max['params'][f'w{i}'] for i in range(len(tickers))])
    best_weights /= best_weights.sum()
    return best_weights

# Get optimal portfolio weights
optimal_weights = optimize_portfolio()

# Explainable AI (XAI) using SHAP
explainer = shap.Explainer(lambda x: np.dot(x, optimal_weights), returns.values)
shap_values = explainer(returns.values)
shap.summary_plot(shap_values, returns.values, feature_names=tickers)

